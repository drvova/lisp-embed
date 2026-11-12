;;;; gemma.lisp — EmbeddingGemma-2 (Gemma-3n text) embeddings in pure Lisp.
;;;; Bidirectional attention (scale 1.0), RMSNorm q/k/v, RoPE, GELU-tanh gated
;;;; MLP, per-layer embeddings (PLE), layer scalars, 512->768 projection,
;;;; mean pooling, L2-normalized output.

(defpackage :gemma
  (:use :cl)
  (:export #:load-model #:embed #:*hidden* #:*pooling*))
(in-package :gemma)

(defparameter *hidden* 512)
(defparameter *layers* 24)
(defparameter *heads* 4)
(defparameter *inter* 2048)
(defparameter *ple-dim* 512)
(defparameter *out-dim* 768)
(defparameter *eps* 1d-6)
(defparameter *rope-theta* 10000d0)
(defparameter *pooling* :mean)
(defparameter *max-pos* 2048)

(defparameter *w-embed* nil)
(defparameter *w-ple-proj* nil)   ; [layers*ple-dim, hidden]
(defparameter *w-ple-norm* nil)   ; [ple-dim]
(defparameter *w-norm* nil)       ; final norm [hidden]
(defparameter *w-out* nil)        ; [out-dim, hidden]
(defparameter *layers-w* nil)
(defparameter *rope* (make-hash-table)) ; head-dim -> (cos sin)
(defparameter *tokenizer* nil)
(defparameter *dump-dir* nil)
(defparameter *debug* nil)

(defun read-json-file (path)
  (when (probe-file path)
    (with-open-file (in path :direction :input :external-format :utf-8)
      (let ((s (make-string (file-length in))))
        (json:parse (subseq s 0 (read-sequence s in)))))))

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
  (declare (type (simple-array single-float (*)) x g)
           (type fixnum off n)
           (optimize (speed 3) (safety 0)))
  (let ((ss 0.0d0))
    (loop for i of-type fixnum below n
          do (incf ss (* (aref x (+ off i)) (aref x (+ off i)))))
    (let ((inv (coerce (/ 1.0d0 (sqrt (+ (/ ss n) *eps*))) 'single-float)))
      (loop for i of-type fixnum below n
            do (setf (aref x (+ off i)) (* (aref x (+ off i)) inv (aref g i)))))))

(defun rmsnorm-noscale (x off n)
  "RMSNorm without a learned scale (v_norm)."
  (declare (type (simple-array single-float (*)) x)
           (type fixnum off n)
           (optimize (speed 3) (safety 0)))
  (let ((ss 0.0d0))
    (loop for i of-type fixnum below n
          do (incf ss (* (aref x (+ off i)) (aref x (+ off i)))))
    (let ((inv (coerce (/ 1.0d0 (sqrt (+ (/ ss n) *eps*))) 'single-float)))
      (loop for i of-type fixnum below n
            do (setf (aref x (+ off i)) (* (aref x (+ off i)) inv))))))

(defun gelu-tanh (x)
  "gelu_pytorch_tanh: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))."
  (let* ((xd (coerce x 'double-float))
         (c (* 0.7978845608028654d0 (+ xd (* 0.044715d0 xd xd xd)))))
    (coerce (* 0.5d0 xd (+ 1.0d0 (tanh c))) 'single-float)))

(defun init-rope (head-dim theta)
  (let* ((half (floor head-dim 2))
         (cos (make-array (* *max-pos* half) :element-type 'single-float))
         (sin (make-array (* *max-pos* half) :element-type 'single-float)))
    (loop for p of-type fixnum below *max-pos*
          do (loop for i of-type fixnum below half
                   do (let* ((freq (/ 1.0d0 (expt theta
                                                (/ (* 2.0d0 i) head-dim))))
                             (ang (* p freq)))
                        (setf (aref cos (+ (* p half) i)) (coerce (cos ang) 'single-float)
                              (aref sin (+ (* p half) i)) (coerce (sin ang) 'single-float)))))
    (list cos sin)))

(defun rope (x off pos heads head-dim tbl)
  (declare (type (simple-array single-float (*)) x)
           (type fixnum off pos heads head-dim)
           (optimize (speed 3) (safety 0)))
  (let* ((cos (first tbl))
         (sin (second tbl))
         (half (floor head-dim 2)))
    (loop for h of-type fixnum below heads
          do (let ((base (+ off (* h head-dim))))
               (loop for i of-type fixnum below half
                     do (let* ((c (aref cos (+ (* pos half) i)))
                               (s (aref sin (+ (* pos half) i)))
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
  (let* ((dir (make-pathname :defaults path :name nil :type nil))
         (cfg (read-json-file (merge-pathnames "config.json" dir)))
         (tc (and cfg (gethash "text_config" cfg))))
    (when tc
      (setf *hidden* (gethash "hidden_size" tc)
            *layers* (gethash "num_hidden_layers" tc)
            *heads* (gethash "num_attention_heads" tc)
            *inter* (gethash "intermediate_size" tc)
            *ple-dim* (or (gethash "hidden_size_per_layer_input" tc) *hidden*)
            *out-dim* (or (gethash "embedding_dim" tc) 768)
            *eps* (coerce (or (gethash "rms_norm_eps" tc) 1d-6) 'double-float)))
    (format *error-output* "gemma config: hidden=~D layers=~D heads=~D ffn=~D ple=~D out=~D~%"
            *hidden* *layers* *heads* *inter* *ple-dim* *out-dim*)
    (multiple-value-bind (header data-start) (st:parse-header path)
      (setf (gethash "data-start" header) data-start)
      (setf *w-embed* (st:get-tensor path header "language_model.embed_tokens.weight")
            *w-ple-proj* (st:get-tensor path header "language_model.ple.per_layer_model_projection.weight")
            *w-ple-norm* (st:get-tensor path header "language_model.ple.per_layer_projection_norm.weight")
            *w-norm* (st:get-tensor path header "language_model.norm.weight")
            *w-out* (st:get-tensor path header "language_model.embedding_projection.weight")
            *layers-w* (make-array *layers*))
      (loop for li below *layers*
            do (let* ((q-norm (st:get-tensor path header
                                             (format nil "language_model.layers.~D.self_attn.q_norm.weight" li)))
                      (k-name (format nil "language_model.layers.~D.self_attn.k_proj.weight" li))
                      (k-w (st:get-tensor path header k-name))
                      (hd (length q-norm))
                      (kv (/ (first (gethash "shape" (gethash k-name header))) hd)))
                 (setf (aref *layers-w* li)
                     (list
                      :hd hd
                      :kv kv
                      :rope (init-rope hd (if (string= (aref (coerce (gethash "layer_types" tc) 'vector) li)
                                                        "full_attention")
                                              1d6 10000d0))
                      :in-ln (st:get-tensor path header
                                           (format nil "language_model.layers.~D.input_layernorm.weight" li))
                      :post-ln (st:get-tensor path header
                                              (format nil "language_model.layers.~D.post_attention_layernorm.weight" li))
                      :pre-ff-ln (st:get-tensor path header
                                                (format nil "language_model.layers.~D.pre_feedforward_layernorm.weight" li))
                      :post-ff-ln (st:get-tensor path header
                                                 (format nil "language_model.layers.~D.post_feedforward_layernorm.weight" li))
                      :q-w (st:get-tensor path header
                                          (format nil "language_model.layers.~D.self_attn.q_proj.weight" li))
                      :k-w (st:get-tensor path header
                                          (format nil "language_model.layers.~D.self_attn.k_proj.weight" li))
                      :v-w (st:get-tensor path header
                                          (format nil "language_model.layers.~D.self_attn.v_proj.weight" li))
                      :o-w (st:get-tensor path header
                                          (format nil "language_model.layers.~D.self_attn.o_proj.weight" li))
                      :q-norm (st:get-tensor path header
                                             (format nil "language_model.layers.~D.self_attn.q_norm.weight" li))
                      :k-norm (st:get-tensor path header
                                             (format nil "language_model.layers.~D.self_attn.k_norm.weight" li))
                      :gate (st:get-tensor path header
                                           (format nil "language_model.layers.~D.mlp.gate_proj.weight" li))
                      :up (st:get-tensor path header
                                         (format nil "language_model.layers.~D.mlp.up_proj.weight" li))
                      :down (st:get-tensor path header
                                           (format nil "language_model.layers.~D.mlp.down_proj.weight" li))
                      :ple-gate (st:get-tensor path header
                                               (format nil "language_model.layers.~D.ple_block.per_layer_input_gate.weight" li))
                      :ple-proj (st:get-tensor path header
                                               (format nil "language_model.layers.~D.ple_block.per_layer_projection.weight" li))
                      :ple-norm (st:get-tensor path header
                                               (format nil "language_model.layers.~D.ple_block.post_per_layer_input_norm.weight" li))
                      :scalar (aref (st:get-tensor path header
                                                    (format nil "language_model.layers.~D.layer_scalar" li)) 0)))))

      (setf *tokenizer* (tokenizer:make-tokenizer-from-json (merge-pathnames "tokenizer.json" dir)))
      (format *error-output* "gemma model loaded: ~D layers~%" *layers*))))

;;; ---- forward pass ----

(defun layer-forward (h ple lw seq q-buf k-buf v-buf ctx scores tmp out normed)
  "One Gemma-3n encoder layer; h updated in place. ple = this layer's [seq x ple-dim]."
  (declare (type (simple-array single-float (*)) h ple q-buf k-buf v-buf ctx scores tmp out normed))
  (let* ((hidden *hidden*)
         (heads *heads*)
         (hd (getf lw :hd))
         (kv (getf lw :kv))
         (qdim (* heads hd))
         (kvdim (* kv hd))
         (inter *inter*)
         (group (floor heads kv)))
    ;; attention sub-block
    (loop for tt of-type fixnum below seq
          do (loop for i of-type fixnum below hidden
                   do (setf (aref normed (+ (* tt hidden) i)) (aref h (+ (* tt hidden) i))))
             (rmsnorm normed (* tt hidden) hidden (getf lw :in-ln)))
    (when *debug* (format t "  normed[0,:3]=~A~%" (coerce (subseq normed 0 3) 'list)))
    (loop for tt of-type fixnum below seq
          do (linear normed (* tt hidden) (getf lw :q-w) q-buf (* tt qdim) hidden qdim)
             (linear normed (* tt hidden) (getf lw :k-w) k-buf (* tt kvdim) hidden kvdim)
             (linear normed (* tt hidden) (getf lw :v-w) v-buf (* tt kvdim) hidden kvdim))
    ;; q/k/v norms per head
    (loop for tt of-type fixnum below seq
          do (loop for hh of-type fixnum below heads
                   do (rmsnorm q-buf (+ (* tt qdim) (* hh hd)) hd (getf lw :q-norm)))
             (loop for hh of-type fixnum below kv
                   do (rmsnorm k-buf (+ (* tt kvdim) (* hh hd)) hd (getf lw :k-norm))
                      (rmsnorm-noscale v-buf (+ (* tt kvdim) (* hh hd)) hd))
             (rope q-buf (* tt qdim) tt heads hd (getf lw :rope))
             (rope k-buf (* tt kvdim) tt kv hd (getf lw :rope)))
    (when *debug* (format t "  q[0,:3]=~A qnorm[0,:3]=~A vnorm[0,:3]=~A~%"
                          (coerce (subseq q-buf 0 3) 'list)
                          (coerce (subseq q-buf 0 3) 'list)
                          (coerce (subseq v-buf 0 3) 'list)))
    ;; bidirectional attention, scaling 1.0
    (loop for hh of-type fixnum below heads
          do (let ((kvh (floor hh group)))
               (loop for t1 of-type fixnum below seq
                     do (loop for t2 of-type fixnum below seq
                              do (setf (aref scores t2)
                                       (%dot q-buf (+ (* t1 qdim) (* hh hd))
                                             k-buf (+ (* t2 kvdim) (* kvh hd)) hd)))
                        (softmax scores 0 seq)
                        (loop for i of-type fixnum below hd
                              do (let ((acc 0.0))
                                   (declare (type single-float acc))
                                   (loop for t2 of-type fixnum below seq
                                         do (incf acc (* (aref scores t2)
                                                         (aref v-buf (+ (* t2 kvdim) (* kvh hd) i)))))
                                   (setf (aref ctx (+ (* t1 qdim) (* hh hd) i)) acc))))))
    (when *debug* (format t "  ctx[0,:3]=~A~%" (coerce (subseq ctx 0 3) 'list)))
    (loop for tt of-type fixnum below seq
          do (linear ctx (* tt qdim) (getf lw :o-w) tmp (* tt hidden) qdim hidden)
             (rmsnorm tmp (* tt hidden) hidden (getf lw :post-ln))
             (loop for i of-type fixnum below hidden
                   do (incf (aref h (+ (* tt hidden) i)) (aref tmp (+ (* tt hidden) i)))))
    ;; FFN sub-block
    (loop for tt of-type fixnum below seq
          do (loop for i of-type fixnum below hidden
                   do (setf (aref normed (+ (* tt hidden) i)) (aref h (+ (* tt hidden) i))))
             (rmsnorm normed (* tt hidden) hidden (getf lw :pre-ff-ln)))
    (loop for tt of-type fixnum below seq
          do (linear normed (* tt hidden) (getf lw :gate) out (* tt inter) hidden inter)
             (linear normed (* tt hidden) (getf lw :up) tmp (* tt inter) hidden inter)
             (loop for i of-type fixnum below inter
                   do (setf (aref out (+ (* tt inter) i))
                            (* (gelu-tanh (aref out (+ (* tt inter) i)))
                               (aref tmp (+ (* tt inter) i)))))
             (linear out (* tt inter) (getf lw :down) tmp (* tt hidden) inter hidden)
             (rmsnorm tmp (* tt hidden) hidden (getf lw :post-ff-ln))
             (loop for i of-type fixnum below hidden
                   do (incf (aref h (+ (* tt hidden) i)) (aref tmp (+ (* tt hidden) i)))))
    ;; PLE sub-block: gate(h) -> act -> * ple -> proj -> norm -> residual
    (loop for tt of-type fixnum below seq
          do (linear h (* tt hidden) (getf lw :ple-gate) out (* tt *ple-dim*) hidden *ple-dim*)
             (loop for i of-type fixnum below *ple-dim*
                   do (setf (aref out (+ (* tt *ple-dim*) i))
                            (* (gelu-tanh (aref out (+ (* tt *ple-dim*) i)))
                               (aref ple (+ (* tt *ple-dim*) i)))))
             (linear out (* tt *ple-dim*) (getf lw :ple-proj) tmp (* tt hidden) *ple-dim* hidden)
             (rmsnorm tmp (* tt hidden) hidden (getf lw :ple-norm))
             (loop for i of-type fixnum below hidden
                   do (incf (aref h (+ (* tt hidden) i)) (aref tmp (+ (* tt hidden) i)))))
    ;; layer scalar
    (let ((sc (getf lw :scalar)))
      (loop for i of-type fixnum below (* seq hidden)
            do (setf (aref h i) (* (aref h i) sc))))
    h))

(defun embed (text)
  "text -> L2-normalized 768-dim embedding (mean pooling + projection)."
  (multiple-value-bind (ids mask) (tokenizer:tk-encode *tokenizer* text)
    (declare (ignore mask))
    (let* ((seq (length ids))
           (hidden *hidden*)
           (heads *heads*)
           (qdim (* heads 512))          ; upper bound for the full-attention layers
           (kvdim 512)
           (h (make-array (* seq hidden) :element-type 'single-float))
           (ple (make-array (* seq *layers* *ple-dim*) :element-type 'single-float))
           (q-buf (make-array (* seq qdim) :element-type 'single-float))
           (k-buf (make-array (* seq kvdim) :element-type 'single-float))
           (v-buf (make-array (* seq kvdim) :element-type 'single-float))
           (ctx (make-array (* seq qdim) :element-type 'single-float))
           (scores (make-array seq :element-type 'single-float))
           (tmp (make-array (* seq (max hidden *ple-dim* *inter*)) :element-type 'single-float))
           (out (make-array (* seq (max *inter* *ple-dim*)) :element-type 'single-float))
           (normed (make-array (* seq hidden) :element-type 'single-float))
           (ple-buf (make-array (* seq *layers* *ple-dim*) :element-type 'single-float))
           (scale (sqrt (coerce hidden 'single-float))))
      ;; token embeddings, scaled by sqrt(hidden)
      (loop for tt of-type fixnum below seq
            do (loop for i of-type fixnum below hidden
                     do (setf (aref h (+ (* tt hidden) i))
                              (* scale (aref *w-embed* (+ (* (aref ids tt) hidden) i))))))
      ;; per-layer embeddings: projection * 1/sqrt(hidden), reshape, RMSNorm
      (loop for tt of-type fixnum below seq
            do (linear h (* tt hidden) *w-ple-proj* ple-buf (* tt *layers* *ple-dim*) hidden (* *layers* *ple-dim*)))
      (let ((n (* seq *layers* *ple-dim*))
            (inv (/ 1.0 (sqrt (coerce hidden 'single-float)))))
        (loop for i of-type fixnum below n
              do (setf (aref ple-buf i) (* (aref ple-buf i) inv))))
      (loop for tt of-type fixnum below seq
            do (loop for li of-type fixnum below *layers*
                     do (loop for i of-type fixnum below *ple-dim*
                              do (setf (aref ple (+ (* tt *layers* *ple-dim*) (* li *ple-dim*) i))
                                       (aref ple-buf (+ (* tt *layers* *ple-dim*) (* li *ple-dim*) i))))
                        (rmsnorm ple (+ (* tt *layers* *ple-dim*) (* li *ple-dim*)) *ple-dim* *w-ple-norm*)))
      ;; layers
      (loop for li of-type fixnum below *layers*
            do (let ((ple-slice (make-array (* seq *ple-dim*) :element-type 'single-float)))
                 (loop for tt of-type fixnum below seq
                       do (loop for i of-type fixnum below *ple-dim*
                                do (setf (aref ple-slice (+ (* tt *ple-dim*) i))
                                         (aref ple (+ (* tt *layers* *ple-dim*) (* li *ple-dim*) i)))))
                 (layer-forward h ple-slice (aref *layers-w* li) seq
                                q-buf k-buf v-buf ctx scores tmp out normed)
                 (when *dump-dir*
                   (with-open-file (s (format nil "~A/h~D.txt" *dump-dir* li)
                                      :direction :output :if-exists :supersede)
                     (loop for i below (* seq hidden)
                           do (format s "~,6F~%" (aref h i)))))))
      ;; final norm
      (loop for tt of-type fixnum below seq
            do (rmsnorm h (* tt hidden) hidden *w-norm*))
      ;; mean pool -> project -> normalize
      (let ((vec (make-array *out-dim* :element-type 'single-float))
            (pooled (make-array hidden :element-type 'single-float))
            (norm 0.0))
        (loop for tt of-type fixnum below seq
              do (loop for i of-type fixnum below hidden
                       do (incf (aref pooled i) (aref h (+ (* tt hidden) i)))))
        (loop for i of-type fixnum below hidden
              do (setf (aref pooled i) (/ (aref pooled i) (coerce seq 'single-float))))
        (linear pooled 0 *w-out* vec 0 hidden *out-dim*)
        (loop for i of-type fixnum below *out-dim*
              do (incf norm (* (aref vec i) (aref vec i))))
        (setf norm (sqrt norm))
        (loop for i of-type fixnum below *out-dim*
              do (setf (aref vec i) (/ (aref vec i) norm)))
        vec))))
