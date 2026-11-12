;;;; embed.lisp — parallel chunk embedder: kb_chunks.jsonl -> flat index.
;;;; Writes index.bin (N x 384 f32), meta.tsv (source<TAB>domain, header
;;;; "N<TAB>384"), texts.tsv (escaped text) — the exact format kb.lisp reads.
;;;;
;;;; Torch-free rebuild path. CPU-bound: the full 2.79M-chunk corpus takes
;;;; days on this box (see README); incremental rebuilds are practical.
;;;;
;;;; Usage:
;;;;   sbcl --dynamic-space-size 8192 --load embed.lisp
;;;;   (embed:run)                 ; full kb_chunks.jsonl
;;;;   (embed:run :limit 1000)     ; first N chunks (smoke test)

(load (merge-pathnames "json.fasl"
                       (or *compile-file-pathname* *load-pathname*)))
(load (merge-pathnames "tokenizer.fasl"
                       (or *compile-file-pathname* *load-pathname*)))
(load (merge-pathnames "bert.fasl"
                       (or *compile-file-pathname* *load-pathname*)))

(defpackage :embed
  (:use :cl)
  (:export #:run))
(in-package :embed)

(defparameter *chunks* #P"D:/cybersec-kb/kb_chunks.jsonl")
(defparameter *index* #P"D:/cybersec-kb/index.bin.new")
(defparameter *meta* #P"D:/cybersec-kb/meta.tsv.new")
(defparameter *texts* #P"D:/cybersec-kb/texts.tsv.new")
(defparameter *threads* 16)
(defparameter *max-chars* (parse-integer (cfg:env "KB_MAX_CHARS" "256")))  ; embedding input cap (chunk text stored in full)
(defparameter *model*
  "C:/Users/bigmo/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors")
(defparameter *model-marker* #P"D:/cybersec-kb/index.model.new")

(defun esc (text)
  "TSV escaping: backslash, tab, newline, CR, non-ASCII -> \\uXXXX."
  (with-output-to-string (out)
    (loop for c across text
          for o = (char-code c)
          do (cond ((char= c #\\) (write-string "\\\\" out))
                   ((char= c #\Tab) (write-string "\\t" out))
                   ((char= c #\Newline) (write-string "\\n" out))
                   ((char= c #\Return) (write-string "\\r" out))
                   ((< o 128) (write-char c out))
                   (t (format out "\\u~4,'0x" o))))))

(defun write-f32 (stream vec)
  "Write a (simple-array single-float (*)) as little-endian f32 bytes."
  (let ((buf (make-array (length vec) :element-type '(unsigned-byte 32))))
    (sb-sys:with-pinned-objects (vec buf)
      (let ((src (sb-alien:sap-alien (sb-sys:vector-sap vec)
                                     (* (sb-alien:unsigned 32))))
            (dst (sb-alien:sap-alien (sb-sys:vector-sap buf)
                                     (* (sb-alien:unsigned 32)))))
        (dotimes (i (length vec))
          (setf (sb-alien:deref dst i) (sb-alien:deref src i)))))
    (write-sequence buf stream)))

(defun count-chunks ()
  "Count non-empty lines in *chunks*."
  (with-open-file (in *chunks* :direction :input :external-format :utf-8)
    (loop for line = (read-line in nil nil)
          while line
          count (plusp (length line)))))

(defun run (&key (limit nil) (threads *threads*))
  (setf *threads* threads)
  (bert:load-model *model*)
  (let ((total (or limit (count-chunks))))
    (format t "model loaded; ~D chunks to embed with ~D threads~%" total *threads*)
    (finish-output)
    (let ((n 0)
          (t0 (get-internal-real-time)))
      (with-open-file (in *chunks* :direction :input :external-format :utf-8)
        (with-open-file (vb *index* :direction :output :if-exists :supersede
                                  :element-type '(unsigned-byte 32))
          (with-open-file (mf *meta* :direction :output :if-exists :supersede
                                    :external-format :utf-8)
            (with-open-file (tf *texts* :direction :output :if-exists :supersede
                                      :external-format :utf-8)
              (format mf "~D~C384~%" total #\Tab)
              (loop
                (let ((batch nil))
                  (dotimes (_ *threads*)
                    (let ((line (read-line in nil nil)))
                      (when line (push line batch))))
                  (when (null batch) (return))
                  (setf batch (nreverse batch))
                  (let ((futures
                          (mapcar (lambda (line)
                                    (let ((text (gethash "text" (json:parse line))))
                                      (let ((short (subseq text 0 (min *max-chars* (length text)))))
                                        (sb-thread:make-thread
                                         (lambda () (bert:embed short))
                                         :name "embed"))))
                                  batch)))
                    (dolist (f futures)
                      (write-f32 vb (sb-thread:join-thread f))
                      (incf n)))
                  (dolist (line batch)
                    (let ((rec (json:parse line)))
                      (format mf "~A~C~A~%"
                              (gethash "source" rec) #\Tab (gethash "domain" rec))
                      (format tf "~A~%" (esc (gethash "text" rec)))))
                  (when (zerop (mod n 160))
                    (format t "  ~D/~D chunks (~,1F/s)~%"
                            n total
                            (/ n (max 1 (/ (- (get-internal-real-time) t0)
                                           internal-time-units-per-second))))
                    (finish-output))
                  (when (>= n total) (return))))))))
      (format t "done: ~D chunks in ~,0Fs~%"
              n (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
      (with-open-file (mf *model-marker* :direction :output :if-exists :supersede)
        (format mf "~A~%" *model*))
      (format t "outputs written as *.new — swap them in manually after verifying:~%  ~A~%  ~A~%  ~A~%  ~A~%"
              *index* *meta* *texts* *model-marker*))))
