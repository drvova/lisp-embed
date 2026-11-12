;;;; llama.lisp — decoder-only embedding models in pure Common Lisp.
;;;; Covers Qwen3-Embedding, Llama, Mistral, Gemma embedding checkpoints:
;;;; RMSNorm, RoPE, grouped-query attention, SwiGLU, causal attention,
;;;; last-token pooling, L2-normalized output.

(defpackage :llama
  (:use :cl)
  (:export #:load-model #:embed #:*hidden* #:*query-prefix* #:*pooling*))
(in-package :llama)

(defparameter *hidden* 1024)
(defparameter *layers* 28)
(defparameter *heads* 16)
(defparameter *kv-heads* 8)
(defparameter *head-dim* 128)
(defparameter *intermediate* 3072)
(defparameter *rope-theta* 1d6)
(defparameter *eps* 1d-6)
(defparameter *pooling* :last)     ; :last | :mean
(defparameter *query-prefix* "")   ; e.g. "Instruct: ...\nQuery: "
(defparameter *max-pos* 1024)

(defparameter *w-embed* nil)
(defparameter *w-norm* nil)
(defparameter *layers-w* nil)
(defparameter *cos* nil)           ; [max-pos x head-dim/2]
(defparameter *sin* nil)
(defparameter *tokenizer* nil)

(defun read-json-file (path)
  (when (probe-file path)
    (with-open-file (in path :direction :input :external-format :utf-8)
      (let ((s (make-string (file-length in))))
        (json:parse (subseq s 0 (read-sequence s in)))))))

(defun tokenizer-specials (dir)
  "Read tokenizer.json: added tokens + TemplateProcessing -> (values bos-id eos-id).
Only the single-sequence template is inspected; a SpecialToken before the
Sequence is BOS, after it is EOS."
  (let ((path (merge-pathnames "tokenizer.json" dir)))
    (when (probe-file path)
      (let* ((tj (read-json-file path))
             (added (gethash "added_tokens" tj))
             (by-name (make-hash-table :test 'equal))
             (bos nil) (eos nil))
        (when added
          (dolist (a added)
            (setf (gethash (gethash "content" a) by-name)
                  (gethash "id" a))))
        (let* ((pp (gethash "post_processor" tj))
               (procs (and pp (gethash "processors" pp)))
               (tp (and procs
                        (find-if (lambda (p) (string= (gethash "type" p) "TemplateProcessing"))
                                 procs))))
          (when tp
            (let ((single (gethash "single" tp))
                  (seen-seq nil))
              (when single
                (dolist (item single)
                  (let ((tok (gethash "SpecialToken" item)))
                    (when tok
                      (let ((id (gethash (gethash "id" tok) by-name)))
                        (if seen-seq
                            (setf eos id)
                            (setf bos id)))))
                  (when (gethash "Sequence" item)
                    (setf seen-seq t)))))))
        (values bos eos)))))

(defun make-model-tokenizer (dir)
  "tokenizer.json when present (authoritative), else vocab.json + merges.txt."
  (let ((tj (merge-pathnames "tokenizer.json" dir))
        (vocab-json (merge-pathnames "vocab.json" dir))
        (merges (merge-pathnames "merges.txt" dir)))
    (cond ((probe-file tj) (tokenizer:make-tokenizer-from-json tj))
          ((and (probe-file vocab-json) (probe-file merges))
           (multiple-value-bind (bos eos) (tokenizer-specials dir)
             (tokenizer:make-tokenizer :bpe :vocab-path vocab-json :merges-path merges
                                       :byte-level t :pretok :llama3 :lowercase nil
                                       :bos-id bos :eos-id eos)))
          (t (error "no tokenizer files in ~A" dir)))))

;;; ---- math ----

(declaim (inline %dot))
(defun %dot (a ai b bi n)
  (declare (type (simple-array single-float (*)) a b)
           (type fixnum ai bi n)
           (optimize (speed 3) (safety 0)))
  (let ((s 0.0))
    (declare (type single-float s))
    (loop for k of-type fixnum below n
          do (incf s (* (aref a (the fixnum (+ ai k)))
                        (aref b (the fixnum (+ bi k))))))
    s))

(defun linear (x x-off w y y-off in-dim out-dim)
  "y[y-off+o] = sum_i x[x-off+i]*W[o*in+i]."
  (declare (type (simple-array single-float (*)) x w y)
           (type fixnum x-off y-off in-dim out-dim)
           (optimize (speed 3) (safety 0)))
  (loop for o of-type fixnum below out-dim
        do (setf (aref y (+ y-off o))
                 (%dot x x-off w (the fixnum (* o in-dim)) in-dim))))

(defun rmsnorm (x off n g)
  "In-place RMSNorm over x[off..off+n)."
  (declare (type (simple-array single-float (*)) x g)
           (type fixnum off n)
           (optimize (speed 3) (safety 0)))
  (let ((ss 0.0d0))
    (loop for i of-type fixnum below n
          do (incf ss (* (aref x (+ off i)) (aref x (+ off i)))))
    (let ((inv (coerce (/ 1.0d0 (sqrt (+ (/ ss n) *eps*))) 'single-float)))
      (loop for i of-type fixnum below n
            do (setf (aref x (+ off i))
                     (* (aref x (+ off i)) inv (aref g i)))))))

(defun silu (x)
  "x * sigmoid(x), evaluated in double to avoid single-float exp overflow."
  (let ((xd (coerce x 'double-float)))
    (coerce (/ xd (+ 1.0d0 (exp (- xd)))) 'single-float)))

(defun init-rope ()
  (let* ((half (floor *head-dim* 2))
         (cos (make-array (* *max-pos* half) :element-type 'single-float))
         (sin (make-array (* *max-pos* half) :element-type 'single-float)))
    (loop for p of-type fixnum below *max-pos*
          do (loop for i of-type fixnum below half
                   do (let* ((freq (/ 1.0d0 (expt *rope-theta*
                                                (/ (* 2.0d0 i) *head-dim*))))
                             (ang (* p freq)))
                        (setf (aref cos (+ (* p half) i)) (coerce (cos ang) 'single-float)
                              (aref sin (+ (* p half) i)) (coerce (sin ang) 'single-float)))))
    (setf *cos* cos *sin* sin)))

(defun rope (x off pos heads)
  "Rotate (x[2i], x[2i+1]) pairs of every head at off, by position pos."
  (declare (type (simple-array single-float (*)) x)
           (type fixnum off pos heads)
           (optimize (speed 3) (safety 0)))
  (let ((half (floor *head-dim* 2)))
    (loop for h of-type fixnum below heads
          do (let ((base (+ off (* h *head-dim*))))
               (loop for i of-type fixnum below half
                     do (let* ((c (aref *cos* (+ (* pos half) i)))
                               (s (aref *sin* (+ (* pos half) i)))
                               (a (aref x (+ base i)))
                               (b (aref x (+ base i half))))
                          (setf (aref x (+ base i)) (- (* a c) (* b s))
                                (aref x (+ base i half)) (+ (* a s) (* b c)))))))))

(defun softmax (arr off n)
  (declare (type (simple-array single-float (*)) arr)
           (type fixnum off n)
           (optimize (speed 3) (safety 0)))
  (let ((maxv most-negative-single-float))
    (declare (type single-float maxv))
    (loop for i of-type fixnum below n
          do (setf maxv (max maxv (aref arr (+ off i)))))
    (let ((sum 0.0))
      (declare (type single-float sum))
      (loop for i of-type fixnum below n
            do (let ((e (exp (- (aref arr (+ off i)) maxv))))
                 (setf (aref arr (+ off i)) e)
                 (incf sum e)))
      (loop for i of-type fixnum below n
            do (setf (aref arr (+ off i)) (/ (aref arr (+ off i)) sum))))))

;;; ---- model loading ----

(defun load-model (path)
  "Load a decoder-only embedding model (Qwen3-Embedding / Llama / Mistral)."
  (let* ((dir (make-pathname :defaults path :name nil :type nil))
         (cfg (read-json-file (merge-pathnames "config.json" dir))))
    (when cfg
      (setf *hidden* (gethash "hidden_size" cfg)
            *layers* (gethash "num_hidden_layers" cfg)
            *heads* (gethash "num_attention_heads" cfg)
            *kv-heads* (or (gethash "num_key_value_heads" cfg)
                           (gethash "num_attention_heads" cfg))
            *head-dim* (or (gethash "head_dim" cfg) (/ *hidden* *heads*))
            *intermediate* (gethash "intermediate_size" cfg)
            *rope-theta* (coerce (or (gethash "rope_theta" cfg) 10000) 'double-float)
            *eps* (coerce (or (gethash "rms_norm_eps" cfg) 1d-6) 'double-float)))
    (format *error-output* "llama config: hidden=~D layers=~D heads=~D kv=~D head-dim=~D ffn=~D~%"
            *hidden* *layers* *heads* *kv-heads* *head-dim* *intermediate*)
    (init-rope)
    (multiple-value-bind (header data-start) (st:parse-header path)
      (setf (gethash "data-start" header) data-start)
      (setf *w-embed* (st:get-tensor path header "embed_tokens.weight")
            *w-norm* (st:get-tensor path header "norm.weight")
            *layers-w* (make-array *layers*))
      (loop for li below *layers*
            do (setf (aref *layers-w* li)
                     (list
                      :in-ln (st:get-tensor path header
                                            (format nil "layers.~D.input_layernorm.weight" li))
                      :post-ln (st:get-tensor path header
                                              (format nil "layers.~D.post_attention_layernorm.weight" li))
                      :q-w (st:get-tensor path header
                                          (format nil "layers.~D.self_attn.q_proj.weight" li))
                      :k-w (st:get-tensor path header
                                          (format nil "layers.~D.self_attn.k_proj.weight" li))
                      :v-w (st:get-tensor path header
                                          (format nil "layers.~D.self_attn.v_proj.weight" li))
                      :o-w (st:get-tensor path header
                                          (format nil "layers.~D.self_attn.o_proj.weight" li))
                      :q-norm (st:get-tensor-opt path header
                                                 (format nil "layers.~D.self_attn.q_norm.weight" li))
                      :k-norm (st:get-tensor-opt path header
                                                 (format nil "layers.~D.self_attn.k_norm.weight" li))
                      :gate (st:get-tensor path header
                                           (format nil "layers.~D.mlp.gate_proj.weight" li))
                      :up (st:get-tensor path header
                                         (format nil "layers.~D.mlp.up_proj.weight" li))
                      :down (st:get-tensor path header
                                           (format nil "layers.~D.mlp.down_proj.weight" li)))))
      (setf *tokenizer* (make-model-tokenizer dir))
      (format *error-output* "llama model loaded: ~D layers~%" *layers*))))

;;; ---- forward pass ----

(defun layer-forward (h normed lw seq q-buf k-buf v-buf ctx scores tmp out)
  "One decoder layer; h is updated in place. normed is scratch (seq x hidden)."
  (declare (type (simple-array single-float (*)) h normed q-buf k-buf v-buf ctx scores tmp out))
  (let* ((hidden *hidden*)
         (heads *heads*)
         (kv *kv-heads*)
         (hd *head-dim*)
         (kv-dim (* kv hd))
         (group (floor heads kv)))
    ;; pre-attention RMSNorm into the scratch buffer (h keeps the residual)
    (loop for tt of-type fixnum below seq
          do (loop for i of-type fixnum below hidden
                   do (setf (aref normed (+ (* tt hidden) i))
                            (aref h (+ (* tt hidden) i))))
             (rmsnorm normed (* tt hidden) hidden (getf lw :in-ln)))
    ;; q/k/v projections
    (loop for tt of-type fixnum below seq
          do (linear normed (* tt hidden) (getf lw :q-w) q-buf (* tt heads hd) hidden (* heads hd))
             (linear normed (* tt hidden) (getf lw :k-w) k-buf (* tt kv-dim) hidden kv-dim)
             (linear normed (* tt hidden) (getf lw :v-w) v-buf (* tt kv-dim) hidden kv-dim))
    ;; QK-norm (Qwen3) + RoPE
    (loop for tt of-type fixnum below seq
          do (when (getf lw :q-norm)
               (loop for hh of-type fixnum below heads
                     do (rmsnorm q-buf (+ (* tt heads hd) (* hh hd)) hd (getf lw :q-norm))))
             (when (getf lw :k-norm)
               (loop for hh of-type fixnum below kv
                     do (rmsnorm k-buf (+ (* tt kv-dim) (* hh hd)) hd (getf lw :k-norm))))
             (rope q-buf (* tt heads hd) tt heads)
             (rope k-buf (* tt kv-dim) tt kv))
    ;; causal grouped-query attention
    (loop for hh of-type fixnum below heads
          do (let ((kvh (floor hh group)))
               (loop for t1 of-type fixnum below seq
                     do (loop for t2 of-type fixnum from 0 to t1
                              do (setf (aref scores t2)
                                       (/ (%dot q-buf (+ (* t1 heads hd) (* hh hd))
                                                k-buf (+ (* t2 kv-dim) (* kvh hd))
                                                hd)
                                          (sqrt (coerce hd 'single-float)))))
                        (softmax scores 0 (1+ t1))
                        (loop for i of-type fixnum below hd
                              do (let ((acc 0.0))
                                   (declare (type single-float acc))
                                   (loop for t2 of-type fixnum from 0 to t1
                                         do (incf acc (* (aref scores t2)
                                                         (aref v-buf (+ (* t2 kv-dim) (* kvh hd) i)))))
                                   (setf (aref ctx (+ (* t1 heads hd) (* hh hd) i)) acc))))))
    ;; attention output projection + residual
    (loop for tt of-type fixnum below seq
          do (linear ctx (* tt heads hd) (getf lw :o-w) tmp (* tt hidden) (* heads hd) hidden)
             (loop for i of-type fixnum below hidden
                   do (incf (aref h (+ (* tt hidden) i)) (aref tmp (+ (* tt hidden) i)))))
    ;; pre-FFN RMSNorm into the scratch buffer
    (loop for tt of-type fixnum below seq
          do (loop for i of-type fixnum below hidden
                   do (setf (aref normed (+ (* tt hidden) i))
                            (aref h (+ (* tt hidden) i))))
             (rmsnorm normed (* tt hidden) hidden (getf lw :post-ln)))
    ;; SwiGLU: down(silu(gate(x)) * up(x)) + residual
    (loop for tt of-type fixnum below seq
          do (linear normed (* tt hidden) (getf lw :gate) out (* tt *intermediate*) hidden *intermediate*)
             (linear normed (* tt hidden) (getf lw :up) tmp (* tt *intermediate*) hidden *intermediate*)
             (loop for i of-type fixnum below *intermediate*
                   do (setf (aref out (+ (* tt *intermediate*) i))
                            (* (silu (aref out (+ (* tt *intermediate*) i)))
                               (aref tmp (+ (* tt *intermediate*) i)))))
             (linear out (* tt *intermediate*) (getf lw :down) tmp (* tt hidden) *intermediate* hidden)
             (loop for i of-type fixnum below hidden
                   do (incf (aref h (+ (* tt hidden) i)) (aref tmp (+ (* tt hidden) i)))))
    h))

(defun embed (text)
  "text -> L2-normalized embedding (last-token or mean pooling)."
  (multiple-value-bind (ids mask) (tokenizer:tk-encode
                                    *tokenizer*
                                    (concatenate 'string *query-prefix* text))
    (declare (ignore mask))
    (let* ((seq (length ids))
           (hidden *hidden*)
           (heads *heads*)
           (kv *kv-heads*)
           (hd *head-dim*)
           (h (make-array (* seq hidden) :element-type 'single-float))
           (q-buf (make-array (* seq heads hd) :element-type 'single-float))
           (k-buf (make-array (* seq kv hd) :element-type 'single-float))
           (v-buf (make-array (* seq kv hd) :element-type 'single-float))
           (ctx (make-array (* seq heads hd) :element-type 'single-float))
           (scores (make-array seq :element-type 'single-float))
           (tmp (make-array (* seq (max hidden *intermediate*)) :element-type 'single-float))
           (normed (make-array (* seq hidden) :element-type 'single-float))
           (out (make-array (* seq *intermediate*) :element-type 'single-float)))
      ;; token embeddings
      (loop for tt of-type fixnum below seq
            do (loop for i of-type fixnum below hidden
                     do (setf (aref h (+ (* tt hidden) i))
                              (aref *w-embed* (+ (* (aref ids tt) hidden) i)))))
      ;; decoder layers
      (loop for li of-type fixnum below *layers*
            do (layer-forward h normed (aref *layers-w* li) seq q-buf k-buf v-buf ctx scores tmp out))
      ;; final norm
      (loop for tt of-type fixnum below seq
            do (rmsnorm h (* tt hidden) hidden *w-norm*))
      ;; pooling + L2 normalize
      (let ((vec (make-array hidden :element-type 'single-float))
            (norm 0.0))
        (if (eq *pooling* :mean)
            (progn
              (loop for tt of-type fixnum below seq
                    do (loop for i of-type fixnum below hidden
                             do (incf (aref vec i) (aref h (+ (* tt hidden) i)))))
              (loop for i of-type fixnum below hidden
                    do (setf (aref vec i) (/ (aref vec i) (coerce seq 'single-float)))))
            (loop for i of-type fixnum below hidden
                  do (setf (aref vec i) (aref h (+ (* (1- seq) hidden) i)))))
        (loop for i of-type fixnum below hidden
              do (incf norm (* (aref vec i) (aref vec i))))
        (setf norm (sqrt norm))
        (loop for i of-type fixnum below hidden
              do (setf (aref vec i) (/ (aref vec i) norm)))
        vec))))
