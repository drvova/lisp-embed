;;;; model.lisp — universal embedding-model facade.
;;;; Dispatches load/embed across every supported family by config.json's
;;;; model_type: BERT/RoBERTa/DistilBERT/MPNet, Llama/Qwen/Mistral/Gemma,
;;;; and Gemma-3n (EmbeddingGemma-2).

(defpackage :model
  (:use :cl)
  (:export #:load-model #:embed #:dim #:family #:path #:*family*))
(in-package :model)

(defparameter *family* nil)   ; :bert | :mpnet | :llama | :gemma
(defparameter *path* nil)

(defun read-json-file (path)
  (when (probe-file path)
    (with-open-file (in path :direction :input :external-format :utf-8)
      (let ((s (make-string (file-length in))))
        (json:parse (subseq s 0 (read-sequence s in)))))))

(defun config-type (path)
  "model_type from config.json; multimodal wrappers expose text_config."
  (let* ((dir (make-pathname :defaults path :name nil :type nil))
         (cfg (read-json-file (merge-pathnames "config.json" dir)))
         (tc (and cfg (gethash "text_config" cfg))))
    (or (and (hash-table-p tc) (gethash "model_type" tc))
        (and cfg (gethash "model_type" cfg)))))

(defun load-model (path)
  "Load any supported embedding model; returns the family keyword."
  (let ((mt (config-type path)))
    (setf *path* path)
    (cond
      ((null mt) (error "no config.json model_type at ~A" path))
      ((member mt '("bert" "roberta" "distilbert" "mpnet") :test #'string=)
       (bert:load-model path)
       (setf *family* (if (string= mt "mpnet") :mpnet :bert)))
      ((member mt '("llama" "qwen2" "qwen3" "mistral" "gemma" "gemma2")
               :test #'string=)
       (llama:load-model path)
       (setf *family* :llama))
      ((member mt '("embedding_gemma2" "embedding_gemma2_text") :test #'string=)
       (gemma:load-model path)
       (setf *family* :gemma))
      (t (error "unsupported model_type ~A" mt)))
    (format *error-output* "model: ~A [~A, dim ~D]~%" path *family* (dim))
    *family*))

(defun embed (text)
  "Embed text with whichever family is loaded."
  (ecase *family*
    (:bert (bert:embed text))
    (:mpnet (bert:embed text))
    (:llama (llama:embed text))
    (:gemma (gemma:embed text))))

(defun dim ()
  "Output embedding dimension of the loaded model."
  (ecase *family*
    (:bert bert:*hidden*)
    (:mpnet bert:*hidden*)
    (:llama llama:*hidden*)
    (:gemma gemma::*out-dim*)))

(defun family () *family*)
(defun path () *path*)
