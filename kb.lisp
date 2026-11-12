;;;; kb.lisp — pure-Lisp cybersec knowledge-base query layer (SBCL).
;;;;
;;;; Owns storage, parallel vector scan, retrieval, injection, prefix building.
;;;; Python is reduced to a query-encoder sidecar (embed_server.py).
;;;;
;;;; Start:  sbcl --dynamic-space-size 8192 --load kb.lisp
;;;; Dependencies (json, st, tokenizer, llama, bert) are loaded by repl.cmd /
;;;; serve.lisp before this file; load them in the same order when embedding.
;;;;   (kb-search "pass the hash detection" 3)
;;;;   (kb-inject "harden SSH" 2 2000)
;;;;   (kb-stats) (kb-domains)
;;;;   (kb-prefix "cloud-security" 5 "prefix.txt")

(defpackage :kb
  (:use :cl)
  (:export #:kb-search #:kb-inject #:kb-stats #:kb-domains #:kb-prefix
           #:load-index #:start-encoder #:shutdown))
(in-package :kb)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defparameter *index-dir* (namestring (cfg:data-dir)))
(defparameter *dim* 384)
(defparameter *vectors* nil)      ; (simple-array single-float (*)) N*dim
(defparameter *n* 0)              ; row count
(defparameter *sources* nil)      ; (simple-array t (*)) strings
(defparameter *domains* nil)
(defparameter *text-offsets* nil) ; (simple-array fixnum (*)) byte offsets
(defparameter *texts-stream* nil) ; open binary stream on texts.tsv
(defparameter *threads* 8)        ; scan worker count

(defun unescape (ascii)
  "Undo export escaping: \\t \\n \\r \\\\ \\uXXXX -> characters."
  (let ((out (make-array (length ascii) :element-type 'character
                                        :fill-pointer 0)))
    (let ((i 0) (len (length ascii)))
      (declare (type fixnum i len))
      (loop while (< i len)
            do (let ((c (char ascii i)))
                 (if (char= c #\\)
                     (let ((n (char ascii (1+ i))))
                       (cond
                         ((char= n #\n) (vector-push-extend #\Newline out)
                                        (incf i 2))
                         ((char= n #\t) (vector-push-extend #\Tab out)
                                        (incf i 2))
                         ((char= n #\r) (vector-push-extend #\Return out)
                                        (incf i 2))
                         ((char= n #\\) (vector-push-extend #\\ out)
                                        (incf i 2))
                         ((char= n #\u)
                          ;; \uXXXX -> code-char
                          (let ((cp (parse-integer ascii :start (+ i 2)
                                                         :end (+ i 6)
                                                         :radix 16)))
                            (vector-push-extend (code-char cp) out)
                            (incf i 6)))
                         (t (vector-push-extend n out) (incf i 2))))
                     (progn (vector-push-extend c out) (incf i)))))
      out)))

(defun split-tab (line)
  "Split on the first tab."
  (let ((pos (position #\Tab line)))
    (if pos
        (values (subseq line 0 pos) (subseq line (1+ pos)))
        (values line ""))))

(defun copy-words (src-u32 dst-f32 base n)
  "Bit-copy n uint32 words from src-u32 into dst-f32 starting at word offset
base. IEEE754 bit patterns land in float-typed memory unchanged."
  (declare (type (simple-array (unsigned-byte 32) (*)) src-u32)
           (type (simple-array single-float (*)) dst-f32)
           (type fixnum base n)
           (optimize (speed 3) (safety 0)))
  (sb-sys:with-pinned-objects (src-u32 dst-f32)
    (let ((src (sb-alien:sap-alien (sb-sys:vector-sap src-u32)
                                   (* (sb-alien:unsigned 32))))
          (dst (sb-alien:sap-alien (sb-sys:vector-sap dst-f32)
                                   (* (sb-alien:unsigned 32)))))
      (dotimes (i n)
        (setf (sb-alien:deref dst (+ base i)) (sb-alien:deref src i))))))

(defun load-index ()
  "Load index.bin + meta.tsv into memory; index texts.tsv by byte offset."
  (with-open-file (mf (merge-pathnames "meta.tsv" *index-dir*)
                      :direction :input :external-format :utf-8)
    (let ((header (read-line mf)))
      (setf *n* (parse-integer header :junk-allowed t)
            *sources* (make-array *n*)
            *domains* (make-array *n*))
      (loop for i of-type fixnum below *n*
            for line = (read-line mf nil nil)
            while line
            do (multiple-value-bind (src dom) (split-tab line)
                 (setf (aref *sources* i) src
                       (aref *domains* i) dom)))))
  (setf *text-offsets* (make-array *n* :element-type 'fixnum))
  ;; offset scan: chunked byte reads, track line starts across chunk bounds
  (with-open-file (tf (merge-pathnames "texts.tsv" *index-dir*)
                      :direction :input :element-type '(unsigned-byte 8))
    (let ((buf (make-array 1048576 :element-type '(unsigned-byte 8)))
          (pos 0)
          (row -1))
      (declare (type fixnum pos row))
      (loop for n-read = (read-sequence buf tf)
            while (> n-read 0)
            do (loop for i of-type fixnum below n-read
                     do (when (= (aref buf i) 10)
                          (incf row)
                          (when (and (>= row 0) (< row *n*))
                            (setf (aref *text-offsets* row) (+ pos i 1)))))
               (incf pos n-read))))
  (format t "texts indexed: ~:d offsets~%" *n*)
  ;; index.bin: little-endian float32 bit patterns -> bit-copy into float array
  (setf *vectors* (make-array (* *n* *dim*) :element-type 'single-float))
  (with-open-file (vf (merge-pathnames "index.bin" *index-dir*)
                      :direction :input :element-type '(unsigned-byte 32))
    (let ((buf (make-array 262144 :element-type '(unsigned-byte 32)))
          (base 0))
      (declare (type fixnum base))
      (loop for n-read = (read-sequence buf vf)
            while (> n-read 0)
            do (copy-words buf *vectors* base n-read)
               (incf base n-read))
      (assert (= base (* *n* *dim*))))
    (format t "index loaded: ~:d rows x ~:d dims (~,1f GB)~%"
            *n* *dim* (/ (* *n* *dim* 4) 1073741824.0)))
  (load-ann))

(defun texts-file () (merge-pathnames "texts.tsv" *index-dir*))

(defun chunk-text (idx)
  "Read chunk text for row idx from texts.tsv via its byte offset."
  (unless *texts-stream*
    (setf *texts-stream* (open (texts-file) :direction :input
                                            :element-type '(unsigned-byte 8))))
  (file-position *texts-stream* (aref *text-offsets* idx))
  (let ((bytes (loop for b = (read-byte *texts-stream* nil nil)
                     while (and b (/= b 10))
                     collect b)))
    (unescape (coerce (mapcar #'code-char bytes) 'string))))

(defun start-encoder ()
  "Load the pure-Lisp BERT inference engine (replaces the python sidecar)."
  (model:load-model (cfg:model-path))
  (format t "encoder ready (pure-Lisp BERT)~%"))

(defun encode (text)
  "Embed text via the in-process Lisp BERT engine."
  (model:embed text))

(declaim (inline %dot))
(defun %dot (base q)
  (declare (type (simple-array single-float (*)) *vectors*)
           (type fixnum base))
  (let ((s 0.0))
    (declare (type single-float s))
    (loop for j of-type fixnum below *dim*
          do (incf s (* (aref *vectors* (the fixnum (+ base j)))
                        (aref q j))))
    s))

(defun local-top-k (q start end k)
  "Brute-force cosine scan over [start,end); returns k (idx . score) pairs.
Vectors are pre-normalized, so dot == cosine. Runs on a worker thread with a
private result set — no locks."
  (let ((scores (make-array k :element-type 'single-float
                              :initial-element most-negative-single-float))
        (idx (make-array k :element-type 'fixnum :initial-element -1)))
    (loop for i of-type fixnum from start below end
          for score of-type single-float = (%dot (the fixnum (* i *dim*)) q)
          do (when (> score (aref scores (1- k)))
               (let ((pos (1- k)))
                 (loop while (and (> pos 0)
                                  (> score (aref scores (1- pos))))
                       do (setf (aref scores pos) (aref scores (1- pos))
                                (aref idx pos) (aref idx (1- pos))
                                pos (1- pos)))
                 (setf (aref scores pos) score
                       (aref idx pos) i))))
    (loop for j below k
          when (>= (aref idx j) 0)
            collect (cons (aref idx j) (aref scores j)))))

(defun top-k-brute (q k)
  "Parallel scan: split rows across worker threads, merge local top-k sets."
  (let* ((per (ceiling *n* *threads*))
         (workers
           (loop for t-idx below *threads*
                 for start = (* t-idx per)
                 for end = (min *n* (* (1+ t-idx) per))
                 when (< start end)
                   collect (sb-thread:make-thread
                            (lambda (s e) (local-top-k q s e k))
                            :arguments (list start end)
                            :name "kb-scan")))
         (locals (mapcar (lambda (w) (sb-thread:join-thread w)) workers))
         (merged (sort (apply #'append locals) #'> :key #'cdr)))
    (subseq merged 0 (min k (length merged)))))

(defun kb-search (query &optional (k 8))
  "Top-k semantic search. Returns list of plists (:source :domain :score :text)."
  (unless *vectors* (load-index))
  (unless *encoder* (start-encoder))
  (let ((q (encode query)))
    (loop for (idx . score) in (top-k q k)
          collect (list :source (aref *sources* idx)
                        :domain (aref *domains* idx)
                        :score score
                        :text (chunk-text idx)))))

(defun kb-inject (query &optional (k 8) (budget 12000))
  "Ready-to-paste RAG context block, char-budgeted."
  (let* ((hits (kb-search query k))
         (parts nil)
         (used 0))
    (loop for hit in hits
          for block = (format nil "[source: ~A | domain: ~A]~%~A"
                              (getf hit :source) (getf hit :domain)
                              (getf hit :text))
          do (when (> (+ used (length block)) budget) (loop-finish))
             (push block parts)
             (incf used (length block)))
    (format nil "~{~A~%~%---~%~%~}" (nreverse parts))))

(defun kb-stats ()
  "Alist of (domain . chunk-count)."
  (let ((counts (make-hash-table :test 'equal :size 64)))
    (loop for i of-type fixnum below *n*
          do (incf (gethash (aref *domains* i) counts 0)))
    (loop for k being the hash-keys of counts
            using (hash-value v)
          collect (cons k v))))

(defun kb-domains ()
  "Sorted list of domain names."
  (sort (mapcar #'car (kb-stats)) #'string<))

(defun kb-prefix (domain &optional (k 40) (out nil))
  "Write a deterministic domain knowledge prefix file for KV-cache prefill."
  (let* ((out-path (or out (concatenate 'string "prefix-" domain ".txt")))
         (rows (let ((acc nil))
                 (loop for i of-type fixnum below *n*
                       do (when (string= (aref *domains* i) domain)
                            (push i acc)
                            (when (>= (length acc) k) (return))))
                 (nreverse acc))))
    (with-open-file (fh out-path :direction :output
                                 :external-format :utf-8
                                 :if-exists :supersede)
      (format fh "You are a senior cybersecurity assistant. Below is the authoritative ~
knowledge base for the '~A' domain. Answer all questions grounded in this ~
material.~%~%" domain)
      (loop for i in rows
            do (format fh "[~A]~%~A~%~%---~%~%" (aref *sources* i)
                       (chunk-text i))))
    (format t "wrote ~A: ~:d chunks~%" out-path (length rows))))

(defun shutdown ()
  "Stop the encoder sidecar and release the texts stream."
  (when *texts-stream*
    (ignore-errors (close *texts-stream*))))

(defun repl-banner ()
  (format t "kb Lisp REPL loaded (~A scan threads). try:~%" *threads*)
  (format t "  (kb-search \"pass the hash\" 2)~%")
  (format t "  (kb-inject \"harden SSH\" 2 2000)~%")
  (format t "  (kb-stats)  (kb-domains)~%")
  (format t "  (kb-prefix \"cloud-security\" 5 \"prefix.txt\")~%")
  (format t "  (eval '(kb-search \"CVE-2023-5877\" 2))  ; eval out of the box~%"))


;;; ---- ANN (IVF) ----

(defparameter *ann-centroids* nil)
(defparameter *ann-counts* nil)
(defparameter *ann-offsets* nil)
(defparameter *ann-lists* nil)
(defparameter *ann-k* 0)
(defparameter *ann-nprobe* 64)

(defun load-ann (&optional (path (merge-pathnames "ann.bin" (cfg:data-dir))))
  "Load ann.bin if present; returns T on success."
  (when (probe-file path)
    (with-open-file (in path :direction :input :element-type '(unsigned-byte 32))
      (let ((magic (read-byte in))
            (k (read-byte in))
            (dim (read-byte in))
            (n (read-byte in)))
        (when (= magic #x414E4E31)
          (setf *ann-k* k)
          (setf *ann-centroids* (make-array (* k dim) :element-type 'single-float))
          (let ((buf (make-array (* k dim) :element-type '(unsigned-byte 32))))
            (read-sequence buf in)
            (sb-sys:with-pinned-objects (buf *ann-centroids*)
              (let ((src (sb-alien:sap-alien (sb-sys:vector-sap buf)
                                             (* (sb-alien:unsigned 32))))
                    (dst (sb-alien:sap-alien (sb-sys:vector-sap *ann-centroids*)
                                             (* (sb-alien:unsigned 32)))))
                (dotimes (i (* k dim))
                  (setf (sb-alien:deref dst i) (sb-alien:deref src i))))))
          (setf *ann-counts* (make-array k :element-type 'fixnum))
          (let ((cbuf (make-array k :element-type '(unsigned-byte 32))))
            (read-sequence cbuf in)
            (loop for c below k do (setf (aref *ann-counts* c) (aref cbuf c))))
          (setf *ann-offsets* (make-array k :element-type 'fixnum))
          (loop for c of-type fixnum below k
                for acc = 0 then (+ acc (aref *ann-counts* (1- c)))
                do (setf (aref *ann-offsets* c) acc))
          (setf *ann-lists* (make-array n :element-type '(unsigned-byte 32)))
          (read-sequence *ann-lists* in)
          (format t "ann loaded: ~D centroids, nprobe ~D~%" k *ann-nprobe*)
          t)))))

(defun local-top-k-over (q list start cnt k)
  "Exact top-k over the list entries [start, start+cnt)."
  (let ((scores (make-array k :element-type 'single-float
                              :initial-element most-negative-single-float))
        (idx (make-array k :element-type 'fixnum :initial-element -1)))
    (loop for j of-type fixnum from start below (+ start cnt)
          for row = (aref list j)
          for score of-type single-float = (%dot (* row *dim*) q)
          do (when (> score (aref scores (1- k)))
               (let ((pos (1- k)))
                 (loop while (and (> pos 0) (> score (aref scores (1- pos))))
                       do (setf (aref scores pos) (aref scores (1- pos))
                                (aref idx pos) (aref idx (1- pos))
                                pos (1- pos)))
                 (setf (aref scores pos) score
                       (aref idx pos) row))))
    (loop for j below k
          when (>= (aref idx j) 0)
            collect (cons (aref idx j) (aref scores j)))))

(declaim (inline dot-arrays))
(defun dot-arrays (a ai b bi n)
  (declare (type (simple-array single-float (*)) a b)
           (type fixnum ai bi n)
           (optimize (speed 3) (safety 0)))
  (let ((s 0.0))
    (declare (type single-float s))
    (loop for k of-type fixnum below n
          do (incf s (* (aref a (the fixnum (+ ai k)))
                        (aref b (the fixnum (+ bi k))))))
    s))

(defun top-k-ivf (q k)
  "IVF query: nearest nprobe centroids, exact re-rank of their lists."
  (let* ((nprobe (min *ann-nprobe* *ann-k*))
         (cands (loop for c of-type fixnum below *ann-k*
                      collect (cons c (dot-arrays *ann-centroids* (* c *dim*) q 0 *dim*))))
         (best (subseq (sort cands #'> :key #'cdr) 0 nprobe))
         (workers
           (loop for (c . score) in best
                 collect (sb-thread:make-thread
                          (lambda (c)
                            (local-top-k-over q *ann-lists*
                                              (aref *ann-offsets* c)
                                              (aref *ann-counts* c) k))
                          :arguments (list c) :name "kb-ivf")))
         (locals (mapcar (lambda (w) (sb-thread:join-thread w)) workers))
         (merged (sort (apply #'append locals) #'> :key #'cdr)))
    (subseq merged 0 (min k (length merged)))))

(defun top-k (q k)
  "ANN dispatch: IVF when ann.bin is loaded, else brute force."
  (if *ann-centroids* (top-k-ivf q k) (top-k-brute q k)))

;;; boot explicitly: (kb:boot) — see start.lisp
(in-package :kb)
