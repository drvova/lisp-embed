;;;; bert.lisp — BERT encoder inference in pure Common Lisp.
;;;; bge-small-en-v1.5: 12 layers, hidden 384, 12 heads, FFN 1536, gelu.
;;;; Embedding = L2-normalized CLS token of the last layer.

(defpackage :bert
  (:use :cl)
  (:export #:load-model #:embed #:*hidden* #:*pooling* #:*query-prefix* #:*causal*))
(in-package :bert)

(defparameter *hidden* 384)
(defparameter *layers* 12)
(defparameter *heads* 12)
(defparameter *head-dim* 32)
(defparameter *intermediate* 1536)
(defparameter *pooling* :cls)      ; :cls | :mean
(defparameter *query-prefix* "")   ; e.g. "query: " for e5-style models
(defparameter *causal* nil)        ; decoder-style causal attention
(defparameter *model-type* "bert") ; bert | roberta (position offset 2)
(defparameter *family* "bert")     ; bert | mpnet | llama
(defparameter *mpnet-rel-bias* nil)
(defparameter *w-word* nil)
(defparameter *w-pos* nil)
(defparameter *w-type* nil)
(defparameter *emb-ln-g* nil)
(defparameter *emb-ln-b* nil)
(defparameter *layers-w* nil)
(defparameter *tokenizer* nil)

(defun read-json-file (path)
  (when (probe-file path)
    (with-open-file (in path :direction :input :external-format :utf-8)
      (let ((s (make-string (file-length in))))
        (json:parse (subseq s 0 (read-sequence s in)))))))


(defun make-model-tokenizer (dir)
  "tokenizer.json when present (authoritative), else vocab.txt -> WordPiece or
vocab.json + merges.txt -> byte-level BPE."
  (let ((tj (merge-pathnames "tokenizer.json" dir))
        (vocab-txt (merge-pathnames "vocab.txt" dir))
        (vocab-json (merge-pathnames "vocab.json" dir))
        (merges (merge-pathnames "merges.txt" dir)))
    (cond ((probe-file tj) (tokenizer:make-tokenizer-from-json tj))
          ((probe-file vocab-txt)
           (tokenizer:make-tokenizer :wordpiece :vocab-path vocab-txt))
          ((and (probe-file vocab-json) (probe-file merges))
           (tokenizer:make-tokenizer
            :bpe :vocab-path vocab-json :merges-path merges :byte-level t
            :lowercase (not (string= *model-type* "roberta"))
            :cls-token (if (string= *model-type* "roberta") "<s>" "[CLS]")
            :sep-token (if (string= *model-type* "roberta") "</s>" "[SEP]")
            :unk-token (if (string= *model-type* "roberta") "<unk>" "[UNK]")))
          (t (error "no tokenizer files in ~A" dir)))))

(defun load-model (path)
  "Load a BERT-family encoder + tokenizer from the model directory.
Architecture (hidden, layers, heads, FFN), pooling mode and query prefix are
read from config.json / 1_Pooling/config.json, so bge-small/base/large,
MiniLM, e5 and similar safetensors models all work."
  (let* ((dir (make-pathname :defaults path :name nil :type nil))
         (cfg (read-json-file (merge-pathnames "config.json" dir)))
         (pool (read-json-file (merge-pathnames "1_Pooling/config.json" dir))))
    (when cfg
      (let ((mt (gethash "model_type" cfg)))
        (cond
          ((member mt '("llama" "qwen2" "qwen3" "mistral" "gemma" "gemma2")
                   :test #'string=)
           (setf *family* "llama")
           (return-from load-model (llama:load-model path)))
          ((string= mt "mpnet")
           (setf *family* "mpnet" *model-type* "mpnet"))
          ((or (null mt) (member mt '("bert" "roberta" "distilbert") :test #'string=))
           (setf *family* "bert" *model-type* (or mt "bert")))
          (t (error "unsupported model_type ~A" mt))))
      (setf *hidden* (gethash "hidden_size" cfg)
            *layers* (gethash "num_hidden_layers" cfg)
            *heads* (gethash "num_attention_heads" cfg)
            *intermediate* (gethash "intermediate_size" cfg)
            *head-dim* (/ *hidden* *heads*)))
    (setf *pooling* :cls)
    (when pool
      (setf *pooling* (if (gethash "pooling_mode_mean_tokens" pool) :mean :cls)))
    (format *error-output* "config: hidden=~D layers=~D heads=~D ffn=~D pooling=~A family=~A~%"
            *hidden* *layers* *heads* *intermediate* *pooling* *family*)
  (when (string= *family* "mpnet")
    (return-from load-model (load-mpnet-weights path)))
  (multiple-value-bind (header data-start) (st:parse-header path)
    (setf (gethash "data-start" header) data-start)
    (setf *w-word* (st:get-tensor path header "embeddings.word_embeddings.weight"))
    (setf *w-pos* (st:get-tensor path header "embeddings.position_embeddings.weight"))
    (setf *w-type* (or (st:get-tensor path header "embeddings.token_type_embeddings.weight")
                       (make-array (* 2 *hidden*) :element-type 'single-float
                                                  :initial-element 0.0)))
    (setf *emb-ln-g* (st:get-tensor path header "embeddings.LayerNorm.weight"))
    (setf *emb-ln-b* (st:get-tensor path header "embeddings.LayerNorm.bias"))
    (setf *layers-w* (make-array *layers*))
    (loop for li below *layers*
          do (setf (aref *layers-w* li)
                   (list
                    :q-w (st:get-tensor path header
                                     (format nil "encoder.layer.~D.attention.self.query.weight" li))
                    :q-b (st:get-tensor path header
                                     (format nil "encoder.layer.~D.attention.self.query.bias" li))
                    :k-w (st:get-tensor path header
                                     (format nil "encoder.layer.~D.attention.self.key.weight" li))
                    :k-b (st:get-tensor path header
                                     (format nil "encoder.layer.~D.attention.self.key.bias" li))
                    :v-w (st:get-tensor path header
                                     (format nil "encoder.layer.~D.attention.self.value.weight" li))
                    :v-b (st:get-tensor path header
                                     (format nil "encoder.layer.~D.attention.self.value.bias" li))
                    :ao-w (st:get-tensor path header
                                      (format nil "encoder.layer.~D.attention.output.dense.weight" li))
                    :ao-b (st:get-tensor path header
                                      (format nil "encoder.layer.~D.attention.output.dense.bias" li))
                    :ao-g (st:get-tensor path header
                                      (format nil "encoder.layer.~D.attention.output.LayerNorm.weight" li))
                    :ao-b2 (st:get-tensor path header
                                       (format nil "encoder.layer.~D.attention.output.LayerNorm.bias" li))
                    :ff-w (st:get-tensor path header
                                      (format nil "encoder.layer.~D.intermediate.dense.weight" li))
                    :ff-b (st:get-tensor path header
                                      (format nil "encoder.layer.~D.intermediate.dense.bias" li))
                    :ow (st:get-tensor path header
                                    (format nil "encoder.layer.~D.output.dense.weight" li))
                    :ob (st:get-tensor path header
                                    (format nil "encoder.layer.~D.output.dense.bias" li))
                    :og (st:get-tensor path header
                                    (format nil "encoder.layer.~D.output.LayerNorm.weight" li))
                    :ob2 (st:get-tensor path header
                                     (format nil "encoder.layer.~D.output.LayerNorm.bias" li)))))
    (setf *tokenizer* (make-model-tokenizer dir))
    (format *error-output* "model loaded: ~D layers, hidden ~D~%" *layers* *hidden*))))

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

;;; ---- optional BLAS fast path (pure Lisp fallback) ----
(defparameter *blas* nil)
(defparameter *blas-path* (cfg:blas-path))

(defun load-blas (&optional (path *blas-path*))
  "Load OpenBLAS and enable the sgemv fast path. Returns T on success."
  (handler-case
      (progn
        (sb-alien:load-shared-object path)
        (sb-alien:define-alien-routine ("scipy_openblas_set_num_threads" %set-blas-threads)
            sb-alien:void (n sb-alien:int))
        (%set-blas-threads 1)  ; one BLAS thread per SBCL worker
        (sb-alien:define-alien-routine ("scipy_cblas_sgemm" %sgemm) sb-alien:void
          (order sb-alien:int) (transa sb-alien:int) (transb sb-alien:int)
          (m sb-alien:int) (n sb-alien:int) (k sb-alien:int)
          (alpha sb-alien:float)
          (a sb-alien:system-area-pointer) (lda sb-alien:int)
          (b sb-alien:system-area-pointer) (ldb sb-alien:int)
          (beta sb-alien:float)
          (c sb-alien:system-area-pointer) (ldc sb-alien:int))
        (sb-alien:define-alien-routine ("scipy_cblas_sgemv" %sgemv) sb-alien:void
          (order sb-alien:int) (trans sb-alien:int)
          (m sb-alien:int) (n sb-alien:int)
          (alpha sb-alien:float)
          (a sb-alien:system-area-pointer) (lda sb-alien:int)
          (x sb-alien:system-area-pointer) (incx sb-alien:int)
          (beta sb-alien:float)
          (y sb-alien:system-area-pointer) (incy sb-alien:int))
        (setf *blas* t))
    (error (e)
      (format *error-output* "BLAS unavailable (~A); using pure Lisp.~%" e)
      nil)))

(defun linear (x x-off w b y y-off in-dim out-dim)
  "y[y-off+o] = b[o] + sum_i x[x-off+i]*W[o*in+i]."
  (declare (type (simple-array single-float (*)) x w b y)
           (type fixnum x-off y-off in-dim out-dim)
           (optimize (speed 3) (safety 0)))
  (if *blas*
      (progn
        (sb-sys:with-pinned-objects (x w y)
          (%sgemv 101 111 out-dim in-dim 1.0
                  (sb-sys:vector-sap w) in-dim
                  (sb-sys:sap+ (sb-sys:vector-sap x) (* 4 x-off)) 1
                  0.0
                  (sb-sys:sap+ (sb-sys:vector-sap y) (* 4 y-off)) 1))
        (loop for o of-type fixnum below out-dim
              do (incf (aref y (+ y-off o)) (aref b o))))
      (loop for o of-type fixnum below out-dim
            do (setf (aref y (+ y-off o))
                     (+ (aref b o)
                        (%dot x x-off w (the fixnum (* o in-dim)) in-dim))))))

(defun layernorm (x off n g b)
  "In-place LayerNorm over x[off..off+n)."
  (declare (type (simple-array single-float (*)) x g b)
           (type fixnum off n)
           (optimize (speed 3) (safety 0)))
  (let ((mean 0.0) (var 0.0))
    (declare (type single-float mean var))
    (loop for i of-type fixnum below n
          do (incf mean (aref x (+ off i))))
    (setf mean (/ mean (coerce n 'single-float)))
    (loop for i of-type fixnum below n
          do (let ((d (- (aref x (+ off i)) mean)))
               (incf var (* d d))))
    (setf var (/ var (coerce n 'single-float)))
    (let ((std (+ (sqrt var) 1e-12)))
      (loop for i of-type fixnum below n
            do (setf (aref x (+ off i))
                     (+ (* (/ (- (aref x (+ off i)) mean) std)
                           (aref g i))
                        (aref b i)))))))

(defun erf-approx (x)
  "Abramowitz-Stegun 7.1.26: erf(x) = 1 - poly(t)*exp(-x^2), t = 1/(1+px).
Coefficients carry alternating signs; max error 1.5e-7."
  (let* ((ax (abs x))
         (t1 (/ 1.0 (+ 1.0 (* 0.3275911 ax))))
         (poly (* t1 (+ 0.254829592
                        (* t1 (+ -0.284496736
                                 (* t1 (+ 1.421413741
                                          (* t1 (+ -1.453152027
                                                   (* t1 1.061405429))))))))))
         (erf-pos (- 1.0 (* poly (exp (- (* ax ax))))))
         (sign (if (minusp x) -1.0 1.0)))
    (* sign erf-pos)))

(defun gelu (x)
  "Exact GELU: 0.5*x*(1+erf(x/sqrt(2)))."
  (* 0.5 x (+ 1.0 (erf-approx (/ x (sqrt 2.0))))))

(defun softmax (arr off n)
  "In-place softmax over arr[off..off+n)."
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

;;; ---- forward pass ----

(defun encoder-layer (h tmp scores ctx q-buf k-buf v-buf seq li)
  "One transformer layer; returns the new hidden states."
  (let ((lw (aref *layers-w* li)))
    ;; Q/K/V projections
    (loop for tt of-type fixnum below seq
          do (linear h (* tt *hidden*) (getf lw :q-w) (getf lw :q-b)
                     q-buf (* tt *hidden*) *hidden* *hidden*)
             (linear h (* tt *hidden*) (getf lw :k-w) (getf lw :k-b)
                     k-buf (* tt *hidden*) *hidden* *hidden*)
             (linear h (* tt *hidden*) (getf lw :v-w) (getf lw :v-b)
                     v-buf (* tt *hidden*) *hidden* *hidden*))
    ;; attention: full bidirectional, per head
    (if *blas*
        (let ((scale (/ 1.0 (sqrt (coerce *head-dim* 'single-float)))))
          (loop for hh below *heads*
                do (sb-sys:with-pinned-objects (q-buf k-buf v-buf scores ctx)
                     ;; scores = scale * Q_h @ K_h^T   [seq x seq]
                     (%sgemm 101 111 112 seq seq *head-dim* scale
                             (sb-sys:sap+ (sb-sys:vector-sap q-buf) (* 4 hh *head-dim*)) *hidden*
                             (sb-sys:sap+ (sb-sys:vector-sap k-buf) (* 4 hh *head-dim*)) *hidden*
                             0.0
                             (sb-sys:vector-sap scores) seq))
                   (when *causal*
                     (loop for t1 of-type fixnum below seq
                           do (loop for t2 of-type fixnum from (1+ t1) below seq
                                    do (setf (aref scores (+ (* t1 seq) t2)) -1e9))))
                   (loop for t1 of-type fixnum below seq
                         do (softmax scores (* t1 seq) seq))
                   (sb-sys:with-pinned-objects (v-buf scores ctx)
                     ;; ctx_h = P @ V_h   [seq x head-dim]
                     (%sgemm 101 111 111 seq *head-dim* seq 1.0
                             (sb-sys:vector-sap scores) seq
                             (sb-sys:sap+ (sb-sys:vector-sap v-buf) (* 4 hh *head-dim*)) *hidden*
                             0.0
                             (sb-sys:sap+ (sb-sys:vector-sap ctx) (* 4 hh *head-dim*)) *hidden*))))
        (loop for hh below *heads*
              do (loop for t1 of-type fixnum below seq
                       do (loop for t2 of-type fixnum below seq
                                do (setf (aref scores (+ (* t1 seq) t2))
                                         (if (and *causal* (> t2 t1))
                                             -1e9
                                             (/ (%dot q-buf
                                                      (+ (* t1 *hidden*) (* hh *head-dim*))
                                                      k-buf
                                                      (+ (* t2 *hidden*) (* hh *head-dim*))
                                                      *head-dim*)
                                                (sqrt (coerce *head-dim* 'single-float)))))))
                 (loop for t1 of-type fixnum below seq
                       do (softmax scores (* t1 seq) seq))
                 (loop for t1 of-type fixnum below seq
                       do (loop for i below *head-dim*
                                do (setf (aref ctx (+ (* t1 *hidden*) (* hh *head-dim*) i))
                                         (loop for t2 of-type fixnum below seq
                                               sum (* (aref scores (+ (* t1 seq) t2))
                                                      (aref v-buf
                                                            (+ (* t2 *hidden*) (* hh *head-dim*) i)))))))))
    ;; attention output projection + residual + LayerNorm
    (loop for tt of-type fixnum below seq
          do (linear ctx (* tt *hidden*) (getf lw :ao-w) (getf lw :ao-b)
                     tmp (* tt *hidden*) *hidden* *hidden*)
             (loop for i below *hidden*
                   do (setf (aref tmp (+ (* tt *hidden*) i))
                            (+ (aref tmp (+ (* tt *hidden*) i))
                               (aref h (+ (* tt *hidden*) i)))))
             (layernorm tmp (* tt *hidden*) *hidden* (getf lw :ao-g) (getf lw :ao-b2)))
    ;; FFN: gelu(tmp @ Wff.T + b) @ Wo.T + b, residual + LayerNorm
    (loop for tt of-type fixnum below seq
          do (let ((ff (make-array *intermediate* :element-type 'single-float)))
               (linear tmp (* tt *hidden*) (getf lw :ff-w) (getf lw :ff-b)
                       ff 0 *hidden* *intermediate*)
               (loop for i below *intermediate*
                     do (setf (aref ff i) (gelu (aref ff i))))
               (linear ff 0 (getf lw :ow) (getf lw :ob)
                       h (* tt *hidden*) *intermediate* *hidden*)
               (loop for i below *hidden*
                     do (setf (aref h (+ (* tt *hidden*) i))
                              (+ (aref h (+ (* tt *hidden*) i))
                                 (aref tmp (+ (* tt *hidden*) i)))))
               (layernorm h (* tt *hidden*) *hidden* (getf lw :og) (getf lw :ob2))))
    h))


(defun embed-bert (text)
  "text -> L2-normalized CLS embedding."
  (multiple-value-bind (ids mask) (tokenizer:tk-encode
                                    *tokenizer*
                                    (concatenate 'string *query-prefix* text))
    (declare (ignore mask))
    (let* ((seq (length ids))
           (h (make-array (* seq *hidden*) :element-type 'single-float))
           (tmp (make-array (* seq *hidden*) :element-type 'single-float))
           (scores (make-array (* seq seq) :element-type 'single-float))
           (ctx (make-array (* seq *hidden*) :element-type 'single-float)))
      (let ((q-buf (make-array (* seq *hidden*) :element-type 'single-float))
            (k-buf (make-array (* seq *hidden*) :element-type 'single-float))
            (v-buf (make-array (* seq *hidden*) :element-type 'single-float)))
      ;; embeddings
      (loop for tt of-type fixnum below seq
            do (loop for i below *hidden*
                     do (setf (aref h (+ (* tt *hidden*) i))
                              (+ (aref *w-word* (+ (* (aref ids tt) *hidden*) i))
                                 (aref *w-pos* (+ (* (+ tt (if (string= *model-type* "roberta") 2 0)) *hidden*) i))
                                 (aref *w-type* i)))))
      ;; per-token LayerNorm (each 384-dim row independently)
      (loop for tt of-type fixnum below seq
            do (layernorm h (* tt *hidden*) *hidden* *emb-ln-g* *emb-ln-b*))
      ;; encoder layers
      (loop for li below *layers*
            do (setq h (encoder-layer h tmp scores ctx q-buf k-buf v-buf seq li)))
      ;; pooling (:cls = token 0, :mean = average over tokens) + L2 normalize
      (let ((out (make-array *hidden* :element-type 'single-float))
            (norm 0.0))
        (if (eq *pooling* :mean)
            (progn
              (loop for tt of-type fixnum below seq
                    do (loop for i below *hidden*
                             do (incf (aref out i)
                                      (aref h (+ (* tt *hidden*) i)))))
              (loop for i below *hidden*
                    do (setf (aref out i) (/ (aref out i) (coerce seq 'single-float)))))
            (loop for i below *hidden*
                  do (setf (aref out i) (aref h i))))
        (loop for i below *hidden*
              do (incf norm (* (aref out i) (aref out i))))
        (setf norm (sqrt norm))
        (loop for i below *hidden*
              do (setf (aref out i) (/ (aref out i) norm)))
        out)))))

;;; ---- MPNet (relative-position attention) ----

(defun mpnet-bucket (i j)
  "T5-style relative position bucket (32 buckets, max distance 128)."
  (let* ((n (- i j))
         (an (abs n))
         (ret (if (minusp n) 16 0)))
    (if (< an 8)
        (+ ret an)
        (+ ret (min 15 (floor (+ 8.0 (* 8.0 (/ (log (/ an 8.0)) (log 16.0))))))))))

(defun linear-b (x x-off w b y y-off in-dim out-dim)
  "y[y-off+o] = b[o] + sum_i x[x-off+i]*W[o*in+i]."
  (declare (type (simple-array single-float (*)) x w b y)
           (type fixnum x-off y-off in-dim out-dim)
           (optimize (speed 3) (safety 0)))
  (loop for o of-type fixnum below out-dim
        do (setf (aref y (+ y-off o))
                 (+ (aref b o)
                    (%dot x x-off w (the fixnum (* o in-dim)) in-dim)))))

(defun load-mpnet-weights (path)
  (multiple-value-bind (header data-start) (st:parse-header path)
    (setf (gethash "data-start" header) data-start)
    (setf *w-word* (st:get-tensor path header "embeddings.word_embeddings.weight")
          *w-pos* (st:get-tensor path header "embeddings.position_embeddings.weight")
          *w-type* nil
          *emb-ln-g* (st:get-tensor path header "embeddings.LayerNorm.weight")
          *emb-ln-b* (st:get-tensor path header "embeddings.LayerNorm.bias")
          *mpnet-rel-bias* (st:get-tensor path header "encoder.relative_attention_bias.weight")
          *layers-w* (make-array *layers*))
    (loop for li below *layers*
          do (setf (aref *layers-w* li)
                   (list
                    :ln-g (st:get-tensor path header
                                         (format nil "encoder.layer.~D.attention.LayerNorm.weight" li))
                    :ln-b (st:get-tensor path header
                                         (format nil "encoder.layer.~D.attention.LayerNorm.bias" li))
                    :q-w (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.q.weight" li))
                    :q-b (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.q.bias" li))
                    :k-w (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.k.weight" li))
                    :k-b (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.k.bias" li))
                    :v-w (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.v.weight" li))
                    :v-b (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.v.bias" li))
                    :o-w (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.o.weight" li))
                    :o-b (st:get-tensor path header
                                        (format nil "encoder.layer.~D.attention.attn.o.bias" li))
                    :ff-w (st:get-tensor path header
                                         (format nil "encoder.layer.~D.intermediate.dense.weight" li))
                    :ff-b (st:get-tensor path header
                                         (format nil "encoder.layer.~D.intermediate.dense.bias" li))
                    :ow (st:get-tensor path header
                                       (format nil "encoder.layer.~D.output.dense.weight" li))
                    :ob (st:get-tensor path header
                                       (format nil "encoder.layer.~D.output.dense.bias" li))
                    :og (st:get-tensor path header
                                       (format nil "encoder.layer.~D.output.LayerNorm.weight" li))
                    :ob2 (st:get-tensor path header
                                        (format nil "encoder.layer.~D.output.LayerNorm.bias" li)))))
    (setf *tokenizer* (make-model-tokenizer (make-pathname :defaults path :name nil :type nil)))
    (format *error-output* "mpnet model loaded: ~D layers~%" *layers*)))

(defun mpnet-layer (h normed lw seq scores ctx q-buf k-buf v-buf tmp out)
  "MPNet layer: attn from raw h, then LayerNorm(o + h), FFN, LayerNorm(dense + attn)."
  (declare (type (simple-array single-float (*)) h normed scores ctx q-buf k-buf v-buf tmp out))
  (let* ((hidden *hidden*)
         (heads *heads*)
         (hd *head-dim*)
         (inter *intermediate*)
         (sqrt-hd (sqrt (coerce hd 'single-float))))
    ;; q/k/v directly from h (no pre-norm)
    (loop for tt of-type fixnum below seq
          do (linear-b h (* tt hidden) (getf lw :q-w) (getf lw :q-b)
                       q-buf (* tt hidden) hidden hidden)
             (linear-b h (* tt hidden) (getf lw :k-w) (getf lw :k-b)
                       k-buf (* tt hidden) hidden hidden)
             (linear-b h (* tt hidden) (getf lw :v-w) (getf lw :v-b)
                       v-buf (* tt hidden) hidden hidden))
    ;; bidirectional attention + relative position bias
    (loop for hh of-type fixnum below heads
          do (loop for t1 of-type fixnum below seq
                   do (loop for t2 of-type fixnum below seq
                            do (setf (aref scores (+ (* t1 seq) t2))
                                     (+ (/ (%dot q-buf (+ (* t1 hidden) (* hh hd))
                                               k-buf (+ (* t2 hidden) (* hh hd)) hd)
                                           sqrt-hd)
                                        (aref *mpnet-rel-bias*
                                              (+ (* (mpnet-bucket t1 t2) heads) hh)))))
                      (softmax scores (* t1 seq) seq))
             (loop for t1 of-type fixnum below seq
                   do (loop for i of-type fixnum below hd
                            do (setf (aref ctx (+ (* t1 hidden) (* hh hd) i))
                                     (loop for t2 of-type fixnum below seq
                                           sum (* (aref scores (+ (* t1 seq) t2))
                                                  (aref v-buf (+ (* t2 hidden) (* hh hd) i))))))))
    ;; o_proj -> tmp
    (loop for tt of-type fixnum below seq
          do (linear-b ctx (* tt hidden) (getf lw :o-w) (getf lw :o-b)
                       tmp (* tt hidden) hidden hidden))
    ;; attention_output = LayerNorm(o + h) -> normed
    (loop for tt of-type fixnum below seq
          do (loop for i of-type fixnum below hidden
                   do (setf (aref normed (+ (* tt hidden) i))
                            (+ (aref tmp (+ (* tt hidden) i))
                               (aref h (+ (* tt hidden) i)))))
             (layernorm normed (* tt hidden) hidden (getf lw :ln-g) (getf lw :ln-b)))
    ;; intermediate dense + gelu
    (loop for tt of-type fixnum below seq
          do (linear-b normed (* tt hidden) (getf lw :ff-w) (getf lw :ff-b)
                       out (* tt inter) hidden inter)
             (loop for i of-type fixnum below inter
                   do (setf (aref out (+ (* tt inter) i)) (gelu (aref out (+ (* tt inter) i))))))
    ;; layer_output = LayerNorm(dense(intermediate) + attention_output) -> h
    (loop for tt of-type fixnum below seq
          do (linear-b out (* tt inter) (getf lw :ow) (getf lw :ob)
                       tmp (* tt hidden) inter hidden)
             (loop for i of-type fixnum below hidden
                   do (setf (aref tmp (+ (* tt hidden) i))
                            (+ (aref tmp (+ (* tt hidden) i))
                               (aref normed (+ (* tt hidden) i)))))
             (layernorm tmp (* tt hidden) hidden (getf lw :og) (getf lw :ob2))
             (loop for i of-type fixnum below hidden
                   do (setf (aref h (+ (* tt hidden) i))
                            (aref tmp (+ (* tt hidden) i)))))
    h))

(defun embed-mpnet (text)
  (multiple-value-bind (ids mask) (tokenizer:tk-encode
                                    *tokenizer*
                                    (concatenate 'string *query-prefix* text))
    (declare (ignore mask))
    (let* ((seq (length ids))
           (hidden *hidden*)
           (heads *heads*)
           (hd *head-dim*)
           (h (make-array (* seq hidden) :element-type 'single-float))
           (normed (make-array (* seq hidden) :element-type 'single-float))
           (scores (make-array (* seq seq) :element-type 'single-float))
           (ctx (make-array (* seq hidden) :element-type 'single-float))
           (q-buf (make-array (* seq hidden) :element-type 'single-float))
           (k-buf (make-array (* seq hidden) :element-type 'single-float))
           (v-buf (make-array (* seq hidden) :element-type 'single-float))
           (tmp (make-array (* seq hidden) :element-type 'single-float))
           (out (make-array (* seq *intermediate*) :element-type 'single-float)))
      (loop for tt of-type fixnum below seq
            do (loop for i of-type fixnum below hidden
                     do (setf (aref h (+ (* tt hidden) i))
                              (+ (aref *w-word* (+ (* (aref ids tt) hidden) i))
                                 (aref *w-pos* (+ (* (+ tt 2) hidden) i))))))
      (loop for tt of-type fixnum below seq
            do (layernorm h (* tt hidden) hidden *emb-ln-g* *emb-ln-b*))
      (loop for li of-type fixnum below *layers*
            do (mpnet-layer h normed (aref *layers-w* li) seq scores ctx
                            q-buf k-buf v-buf tmp out))
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
                  do (setf (aref vec i) (aref h i))))
        (loop for i of-type fixnum below hidden
              do (incf norm (* (aref vec i) (aref vec i))))
        (setf norm (sqrt norm))
        (loop for i of-type fixnum below hidden
              do (setf (aref vec i) (/ (aref vec i) norm)))
        vec))))

(defun embed (text)
  "text -> L2-normalized embedding, dispatched on the loaded model family."
  (cond ((string= *family* "llama") (llama:embed text))
        ((string= *family* "mpnet") (embed-mpnet text))
        (t (embed-bert text))))
