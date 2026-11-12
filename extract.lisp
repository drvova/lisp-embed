;;;; extract.lisp — dataset -> deduplicated records (kb_records.jsonl).
;;;; Pure Common Lisp port of the original extract.py.
;;;;
;;;; File listing uses `cmd /s /b` (SBCL's recursive DIRECTORY is very slow on
;;;; this tree); parsing uses the index-based JSON reader.

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :sb-md5)
  (load (merge-pathnames "json.fasl"
                         (or *compile-file-pathname* *load-pathname*))))

(defpackage :extract
  (:use :cl)
  (:export #:run))
(in-package :extract)

(defparameter *src* (cfg:path (cfg:env "KB_SRC" "./corpus/")))
(defparameter *out* (merge-pathnames "kb_records.jsonl" (cfg:data-dir)))
(defparameter *seen* nil)

(defun md5-hex (text)
  (let ((digest (sb-md5:md5sum-string text)))
    (if (stringp digest)
        digest
        (with-output-to-string (out)
          (loop for b across digest do (format out "~2,'0x" b))))))

(defun split-lines (text)
  (let ((lines nil) (start 0) (len (length text)))
    (loop for i of-type fixnum below len
          do (when (member (char text i) '(#\Newline #\Return))
               (when (> i start) (push (subseq text start i) lines))
               (setf start (1+ i))))
    (when (> len start) (push (subseq text start) lines))
    (nreverse lines)))

(defun list-files (dir)
  "Fast recursive file listing via cmd dir /s /b, filtered."
  (let* ((out (make-string-output-stream))
         (win-dir (substitute #\\ #\/ (namestring dir)))
         (proc (sb-ext:run-program
                "cmd" (list "/c" "dir" "/s" "/b" "/a-d" win-dir)
                :search t :output out :error nil :wait t)))
    (declare (ignore proc))
    (remove-if (lambda (l)
                 (or (zerop (length l))
                     (search ".cache" l)
                     (search ".metadata" l)
                     (search ".gz" l)))
               (split-lines (get-output-stream-string out)))))

(defun getstr (ht key)
  "String value of KEY in hash-table HT, or NIL."
  (let ((v (and ht (gethash key ht))))
    (and (stringp v) v)))

(defun render-conversations (rec)
  (let ((turns (gethash "conversations" rec))
        (parts nil))
    (dolist (turn turns)
      (let* ((who (or (getstr turn "from") ""))
             (role (cond ((string= who "human") "User")
                         ((string= who "gpt") "Assistant")
                         ((string= who "system") "System")
                         (t "User")))
             (val (or (getstr turn "value") "")))
        (push (format nil "### ~A~%~A" role val) parts)))
    (format nil "~{~A~^~%~%~}" (nreverse parts))))

(defun render-messages (rec)
  (let ((msgs (gethash "messages" rec))
        (parts nil))
    (dolist (m msgs)
      (let ((role (or (getstr m "role") "user"))
            (content (or (getstr m "content") "")))
        (push (format nil "### ~@:(~A~)~%~A" role content) parts)))
    (format nil "~{~A~^~%~%~}" (nreverse parts))))

(defun render-instruction (rec)
  (let ((parts nil))
    (let ((sys (getstr rec "system")))
      (when sys (push (format nil "### System~%~A" sys) parts)))
    (let ((ins (getstr rec "instruction")))
      (when ins (push (format nil "### Instruction~%~A" ins) parts)))
    (let ((inp (getstr rec "input")))
      (when inp (push (format nil "### Input~%~A" inp) parts)))
    (let ((outp (getstr rec "output")))
      (when outp (push (format nil "### Response~%~A" outp) parts)))
    (format nil "~{~A~^~%~%~}" (nreverse parts))))

(defun route (rec)
  "Render one parsed JSON record to text, or NIL."
  (cond ((gethash "conversations" rec) (render-conversations rec))
        ((gethash "messages" rec) (render-messages rec))
        ((or (getstr rec "instruction")
             (and (getstr rec "output")
                  (or (getstr rec "input") (getstr rec "system"))))
         (render-instruction rec))
        ((getstr rec "text") (getstr rec "text"))
        (t
         (with-output-to-string (out)
           (maphash (lambda (k v)
                      (cond ((stringp v) (format out "~A: ~A~%" k v))
                            ((hash-table-p v)
                             (format out "~A: ~A~%" k (json:write-json v)))
                            ((listp v)
                             (format out "~A: ~A~%" k (json:write-json v)))
                            (t (format out "~A: ~A~%" k v))))
                    rec)))))

(defun emit (text source domain kind out)
  "Write one deduped record; returns T if written."
  (when (and text (> (length text) 20))
    (let ((id (md5-hex text)))
      (unless (gethash id *seen*)
        (setf (gethash id *seen*) t)
        (format out "~A~%" (json:write-json
                            (let ((h (make-hash-table :test 'equal)))
                              (setf (gethash "doc_id" h) id
                                    (gethash "source" h) source
                                    (gethash "domain" h) domain
                                    (gethash "kind" h) kind
                                    (gethash "text" h) text)
                              h)))
        t))))

(defun process-jsonl (path rel domain out)
  (with-open-file (in path :direction :input :external-format :utf-8)
    (loop for line = (read-line in nil nil)
          while line
          do (let ((trimmed (string-trim '(#\Return #\Space #\Tab) line)))
               (when (> (length trimmed) 1)
                 (handler-case
                     (let ((rec (json:parse trimmed)))
                       (when (hash-table-p rec)
                         (emit (route rec) rel domain "record" out)))
                   (error () nil)))))))

(defun process-json (path rel domain out)
  (handler-case
      (with-open-file (in path :direction :input :external-format :utf-8)
        (let* ((content (make-string (file-length in)))
               (n (read-sequence content in)))
          (let ((data (json:parse (subseq content 0 n))))
            (dolist (rec (if (listp data) data (list data)))
              (when (hash-table-p rec)
                (emit (route rec) rel domain "record" out))))))
    (error () nil)))

(defun process-text (path rel domain out)
  (handler-case
      (with-open-file (in path :direction :input :external-format :utf-8)
        (let* ((content (make-string (file-length in)))
               (n (read-sequence content in)))
          (emit (subseq content 0 n) rel domain "note" out)))
    (error () nil)))

(defun path-domain (path)
  "Domain = the first directory below the dataset root."
  (let* ((dirs (pathname-directory (pathname path)))
         (tail (member-if (lambda (d) (string= (string d) "cyber-security")) dirs)))
    (if (and tail (cdr tail))
        (string-downcase (string (cadr tail)))
        "root")))

(defun run (&optional (src *src*))
  (setf *src* (let ((s (namestring src)))
                (pathname (if (char= (char s (1- (length s))) #\/)
                              s
                              (concatenate 'string s "/"))))
        *seen* (make-hash-table :test 'equal))
  (let ((files (sort (list-files *src*) #'string<)))
    (format t "files: ~D~%" (length files))
    (finish-output)
    (with-open-file (out *out* :direction :output :if-exists :supersede
                               :external-format :utf-8)
      (loop for path in files
            for i from 1
            for type = (string-downcase (or (pathname-type (pathname path)) ""))
            for rel = (enough-namestring (pathname path) *src*)
            for domain = (path-domain path)
            do (format t "  [~D/~D] ~A~%" i (length files) rel)
               (finish-output)
               (cond ((string= type "jsonl") (process-jsonl path rel domain out))
                     ((string= type "json")
                      (if (> (with-open-file (f path) (file-length f)) 100000000)
                          (format t "    skip large json (already indexed)~%")
                          (process-json path rel domain out)))
                     ((or (string= type "txt") (string= type "md"))
                      (process-text path rel domain out)))))
    (format t "done: ~D unique docs -> ~A~%"
            (hash-table-count *seen*) *out*)))
