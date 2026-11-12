;;;; config.lisp — paths and model selection, overridable via environment.
;;;; KB_DATA_DIR  data directory (index.bin, meta.tsv, texts.tsv, ann.bin, *.jsonl)
;;;; KB_MODEL     path to a model.safetensors directory or file

(defpackage :cfg
  (:use :cl)
  (:export #:env #:data-dir #:model-path #:path))
(in-package :cfg)

(defun env (name default)
  (or (sb-ext:posix-getenv name) default))

(defun path (string)
  (pathname string))

(defun data-dir ()
  "Data directory; defaults to ./data relative to the current directory."
  (let ((d (env "KB_DATA_DIR" "./data/")))
    (if (or (char= (char d (1- (length d))) #\/)
            (char= (char d (1- (length d))) #\\))
        (path d)
        (path (concatenate 'string d "/")))))

(defun model-path ()
  "model.safetensors path or model directory; defaults to all-MiniLM-L6-v2
in the HuggingFace cache."
  (env "KB_MODEL"
       (concatenate 'string
                    (or (sb-ext:posix-getenv "HF_HOME")
                        (concatenate 'string (sb-ext:posix-getenv "USERPROFILE")
                                     "/.cache/huggingface"))
                    "/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
                    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors")))
