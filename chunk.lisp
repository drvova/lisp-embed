;;;; chunk.lisp — records -> retrieval chunks (kb_chunks.jsonl).
;;;; Pure Common Lisp port of chunk.py: paragraph-aligned windows,
;;;; 1800-char max, 200-char overlap, hard-split for unbreakable runs.

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load (merge-pathnames "json.fasl"
                         (or *compile-file-pathname* *load-pathname*))))

(defpackage :chunk
  (:use :cl)
  (:export #:run))
(in-package :chunk)

(defparameter *records* (merge-pathnames "kb_records.jsonl" (cfg:data-dir)))
(defparameter *out* (merge-pathnames "kb_chunks.jsonl" (cfg:data-dir)))
(defparameter *max-chars* 1800)
(defparameter *overlap* 200)

(defun split-text (text max-chars overlap)
  "Paragraph-aligned windows; hard fallback for unbreakable runs."
  (if (<= (length text) max-chars)
      (list text)
      (let ((chunks nil)
            (buf ""))
        (labels ((flush ()
                   (when (> (length buf) 0)
                     (push buf chunks)
                     (setf buf (if (> overlap 0)
                                   (subseq buf (max 0 (- (length buf) overlap)))
                                   "")))))
          (let ((start 0))
            (loop
              (let ((pos (search (format nil "~C~C" #\Newline #\Newline)
                                 text :start2 start)))
                (let ((para (if pos
                                (subseq text start (min (length text) (+ pos 2)))
                                (subseq text start))))
                  (when (> (length para) max-chars)
                    ;; unbreakable paragraph: hard split
                    (flush)
                    (loop for i from 0 below (length para) by (- max-chars overlap)
                          do (push (subseq para i (min (length para) (+ i max-chars)))
                                   chunks))
                    (setf buf ""))
                  (when (<= (length para) max-chars)
                    (when (> (+ (length buf) (length para)) max-chars)
                      (flush))
                    (setf buf (if (> (length buf) 0)
                                  (concatenate 'string buf para)
                                  para)))
                  (if pos
                      (setf start (+ pos 2))
                      (return))))
              (when (>= start (length text)) (return))))
          (when (> (length buf) 0) (push buf chunks)))
        (nreverse chunks))))

(defun run ()
  (let ((n-docs 0) (n-chunks 0))
    (with-open-file (in *records* :direction :input :external-format :utf-8)
      (with-open-file (out *out* :direction :output :if-exists :supersede
                                :external-format :utf-8)
        (loop for line = (read-line in nil nil)
              while line
              do (let ((doc (json:parse line)))
                   (incf n-docs)
                   (let ((text (gethash "text" doc))
                         (ord 0))
                     (dolist (piece (split-text text *max-chars* *overlap*))
                       (let ((h (make-hash-table :test 'equal)))
                         (setf (gethash "chunk_id" h)
                               (format nil "~A:~D" (gethash "doc_id" doc) ord)
                               (gethash "doc_id" h) (gethash "doc_id" doc)
                               (gethash "source" h) (gethash "source" doc)
                               (gethash "domain" h) (gethash "domain" doc)
                               (gethash "kind" h) (gethash "kind" doc)
                               (gethash "text" h) piece)
                         (format out "~A~%" (json:write-json h))
                         (incf n-chunks) (incf ord))))
                   (when (zerop (mod n-docs 20000))
                     (format t "  ~D docs, ~D chunks~%" n-docs n-chunks))))))
    (format t "done: ~D docs -> ~D chunks -> ~A~%" n-docs n-chunks *out*)))

