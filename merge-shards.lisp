;;;; merge-shards.lisp — concatenate shard outputs into one flat store.
;;;;   sbcl --load merge-shards.lisp --eval '(merge-shards:run "shards/" "out/")'
(load (merge-pathnames "json.fasl" *load-pathname*))

(defpackage :merge-shards
  (:use :cl)
  (:export #:run))
(in-package :merge-shards)

(defun copy-bytes (in out &optional n)
  "Copy the stream (or exactly N bytes) from IN to OUT."
  (let ((buf (make-array 1048576 :element-type '(unsigned-byte 8)))
        (left n))
    (loop for got = (read-sequence buf in :end (if left (min left (length buf)) (length buf)))
          while (plusp got)
          do (write-sequence buf out :end got)
             (when left (decf left got))
             (when (and left (zerop left)) (return)))))

(defun shard-dirs (dir)
  (sort (remove-if-not (lambda (p) (and (null (pathname-name p))
                                        (null (pathname-type p))))
                       (directory (merge-pathnames "*/" (pathname dir))))
        #'string< :key #'namestring))

(defun shard-count (d)
  "Chunk count from a shard's meta.tsv header."
  (let ((meta (merge-pathnames "meta.tsv" d)))
    (if (probe-file meta)
        (with-open-file (in meta :external-format :utf-8)
          (let ((header (read-line in)))
            (parse-integer (subseq header 0 (position #\Tab header)))))
        0)))

(defun esc (text)
  "TSV escaping: backslash, tab, newline, CR, non-ASCII -> \\uXXXX."
  (with-output-to-string (out)
    (loop for c across text
          for o = (char-code c)
          do (cond ((char= c #\\) (write-string "\\" out))
                   ((char= c #\Tab) (write-string "\\t" out))
                   ((char= c #\Newline) (write-string "\\n" out))
                   ((char= c #\Return) (write-string "\\r" out))
                   ((< o 128) (write-char c out))
                   (t (format out "\\u~4,'0x" o))))))

(defun write-texts-from-corpus (chunks-path out)
  "Regenerate texts.tsv from the chunk corpus (authoritative, escaped)."
  (with-open-file (in chunks-path :direction :input :external-format :utf-8)
    (loop for line = (read-line in nil nil)
          while line
          do (write-line (esc (gethash "text" (json:parse line))) out))))

(defun run (shards-dir out-dir &key texts-from)
  "Concatenate shard-N/{index.bin,meta.tsv,texts.tsv,index.model} into OUT-DIR."
  (let* ((dirs (shard-dirs shards-dir))
         (total (reduce #'+ dirs :key #'shard-count))
         (model nil))
    (ensure-directories-exist (merge-pathnames "x" (pathname out-dir)))
    (with-open-file (ix (merge-pathnames "index.bin" (pathname out-dir))
                        :direction :output :if-exists :supersede
                        :element-type '(unsigned-byte 8))
      (with-open-file (mx (merge-pathnames "meta.tsv" (pathname out-dir))
                          :direction :output :if-exists :supersede
                          :external-format :utf-8)
        (with-open-file (tx (merge-pathnames "texts.tsv" (pathname out-dir))
                            :direction :output :if-exists :supersede
                            :external-format :utf-8)
          (format mx "~D~C384~%" total #\Tab)
          (when texts-from
            (write-texts-from-corpus texts-from tx))
          (dolist (d dirs)
            (let ((idx (merge-pathnames "index.bin" d))
                  (meta (merge-pathnames "meta.tsv" d))
                  (texts (merge-pathnames "texts.tsv" d))
                  (marker (merge-pathnames "index.model" d)))
              (when (probe-file idx)
                (with-open-file (in idx :element-type '(unsigned-byte 8))
                  (copy-bytes in ix (* (shard-count d) 384 4)))
                (with-open-file (in meta :external-format :utf-8)
                  (read-line in)                       ; header
                  (loop for line = (read-line in nil nil)
                        for k from 0 below (shard-count d)
                        while line do (write-line line mx)))
                (unless texts-from
                  (with-open-file (in texts :external-format :utf-8)
                    (loop for line = (read-line in nil nil)
                          while line do (write-line line tx))))
                (when (and (null model) (probe-file marker))
                  (setf model (string-trim '(#\Space #\Newline #\Return)
                                           (with-open-file (in marker)
                                             (let ((s (make-string (file-length in))))
                                               (subseq s 0 (read-sequence s in)))))))))))))
    (when model
      (with-open-file (mf (merge-pathnames "index.model" (pathname out-dir))
                          :direction :output :if-exists :supersede)
        (format mf "~A~%" model)))
    (format t "merged ~D shards, ~D chunks~%" (length dirs) total)))
