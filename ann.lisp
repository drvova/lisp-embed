;;;; ann.lisp — IVF (inverted file) ANN index in pure Common Lisp.
;;;;
;;;; k-means over a sample of the flat vectors, assign every row to its
;;;; nearest centroid, write ann.bin:
;;;;   u32 magic "ANN1", u32 k, u32 dim, u32 n,
;;;;   k*dim f32 centroids, k u32 counts, n u32 row ids (grouped by centroid).
;;;;
;;;; Usage:  sbcl --dynamic-space-size 8192 --load ann.lisp
;;;;         (ann:run)            ; k=1024, 10 iters, 100k sample
;;;;         (ann:run :k 2048)

(defpackage :ann
  (:use :cl)
  (:export #:run))
(in-package :ann)

(defparameter *index* (merge-pathnames "index.bin" (cfg:data-dir)))
(defparameter *ann* (merge-pathnames "ann.bin" (cfg:data-dir)))
(defparameter *dim* 384)
(defparameter *k* 1024)
(defparameter *sample* 100000)
(defparameter *iters* 10)
(defparameter *threads* 8)

(declaim (inline dot))
(defun dot (a ai b bi n)
  (declare (type (simple-array single-float (*)) a b)
           (type fixnum ai bi n)
           (optimize (speed 3) (safety 0)))
  (let ((s 0.0))
    (declare (type single-float s))
    (loop for k of-type fixnum below n
          do (incf s (* (aref a (the fixnum (+ ai k)))
                        (aref b (the fixnum (+ bi k))))))
    s))

(defun load-vectors ()
  "Read index.bin -> (values vectors n)."
  (with-open-file (in *index* :direction :input :element-type '(unsigned-byte 32))
    (let* ((bytes (file-length in))
           (n (/ bytes *dim*))
           (vecs (make-array (* n *dim*) :element-type 'single-float))
           (buf (make-array 262144 :element-type '(unsigned-byte 32)))
           (pos 0))
      (loop for got = (read-sequence buf in)
            while (plusp got)
            do (sb-sys:with-pinned-objects (buf vecs)
                 (let ((src (sb-alien:sap-alien (sb-sys:vector-sap buf)
                                                (* (sb-alien:unsigned 32))))
                       (dst (sb-alien:sap-alien (sb-sys:vector-sap vecs)
                                                (* (sb-alien:unsigned 32)))))
                   (dotimes (i got)
                     (setf (sb-alien:deref dst (+ pos i))
                           (sb-alien:deref src i)))))
               (incf pos got))
      (values vecs n))))

(defun random-sample (n count)
  "COUNT distinct random row indices below N."
  (let ((seen (make-hash-table))
        (out nil))
    (loop while (< (length out) count)
          do (let ((i (random n)))
               (unless (gethash i seen)
                 (setf (gethash i seen) t)
                 (push i out))))
    (coerce out 'vector)))

(defun nearest-centroid (vecs vi centroids k dim)
  "Index of the centroid closest to row VI."
  (let ((best 0) (best-score most-negative-single-float))
    (declare (type single-float best-score))
    (loop for c of-type fixnum below k
          for score of-type single-float = (dot vecs (* vi dim) centroids (* c dim) dim)
          do (when (> score best-score)
               (setf best-score score best c)))
    best))

(defun kmeans (vecs sample k dim iters)
  "Train k centroids on the sampled rows; returns (simple-array single-float (*))."
  (let ((centroids (make-array (* k dim) :element-type 'single-float)))
    ;; init: copy k random sample rows
    (dotimes (c k)
      (let ((row (aref sample c)))
        (replace centroids vecs :start1 (* c dim) :start2 (* row dim)
                 :end2 (+ (* row dim) dim))))
    (let ((assign (make-array (length sample) :element-type 'fixnum)))
      (dotimes (iter iters)
        ;; assign sample rows to centroids (parallel)
        (let* ((n (length sample))
               (per (ceiling n *threads*))
               (workers
                 (loop for t-idx below *threads*
                       for s = (* t-idx per)
                       for e = (min n (* (1+ t-idx) per))
                       when (< s e)
                         collect (sb-thread:make-thread
                                  (lambda (s e)
                                    (loop for j of-type fixnum from s below e
                                          do (setf (aref assign j)
                                                   (nearest-centroid
                                                    vecs (aref sample j)
                                                    centroids k dim))))
                                  :arguments (list s e)))))
          (mapc #'sb-thread:join-thread workers))
        ;; update centroids
        (let ((sums (make-array (* k dim) :element-type 'single-float))
              (counts (make-array k :element-type 'fixnum)))
          (loop for j of-type fixnum below (length sample)
                for row = (aref sample j)
                for c = (aref assign j)
                do (incf (aref counts c))
                   (loop for d of-type fixnum below dim
                         do (incf (aref sums (+ (* c dim) d))
                                  (aref vecs (+ (* row dim) d)))))
          (dotimes (c k)
            (let ((cnt (aref counts c)))
              (when (plusp cnt)
                (loop for d of-type fixnum below dim
                      do (setf (aref centroids (+ (* c dim) d))
                               (/ (aref sums (+ (* c dim) d))
                                  (coerce cnt 'single-float))))))))
        (format t "  kmeans iter ~D/~D~%" (1+ iter) iters)
        (finish-output)))
    centroids))

(defun assign-all (vecs n centroids k dim)
  "Assign every row; returns (values counts lists) where lists is a flat
u32 array of row ids grouped by centroid."
  (let ((assign (make-array n :element-type 'fixnum)))
    (let* ((per (ceiling n *threads*))
           (workers
             (loop for t-idx below *threads*
                   for s = (* t-idx per)
                   for e = (min n (* (1+ t-idx) per))
                   when (< s e)
                     collect (sb-thread:make-thread
                              (lambda (s e)
                                (loop for i of-type fixnum from s below e
                                      do (setf (aref assign i)
                                               (nearest-centroid
                                                vecs i centroids k dim))))
                              :arguments (list s e)))))
      (mapc #'sb-thread:join-thread workers))
    (let ((counts (make-array k :element-type 'fixnum)))
      (loop for i of-type fixnum below n
            do (incf (aref counts (aref assign i))))
      (let* ((offsets (make-array k :element-type 'fixnum))
             (lists (make-array n :element-type '(unsigned-byte 32))))
        (loop for c of-type fixnum below k
              for acc = 0 then (+ acc (aref counts (1- c)))
              do (setf (aref offsets c) acc))
        (let ((cursor (copy-seq offsets)))
          (loop for i of-type fixnum below n
                for c = (aref assign i)
                do (setf (aref lists (aref cursor c)) i)
                   (incf (aref cursor c))))
        (values counts lists)))))

(defun write-u32 (stream value)
  (let ((buf (make-array 1 :element-type '(unsigned-byte 32))))
    (setf (aref buf 0) value)
    (write-sequence buf stream)))

(defun save-ann (centroids counts lists n k dim)
  (with-open-file (out *ann* :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 32))
    (let ((hdr (make-array 4 :element-type '(unsigned-byte 32))))
      (setf (aref hdr 0) #x414E4E31  ; "ANN1"
            (aref hdr 1) k
            (aref hdr 2) dim
            (aref hdr 3) n)
      (write-sequence hdr out))
    ;; centroids as raw f32 (bit-copied from the float array)
    (let ((buf (make-array (length centroids) :element-type '(unsigned-byte 32))))
      (sb-sys:with-pinned-objects (centroids buf)
        (let ((src (sb-alien:sap-alien (sb-sys:vector-sap centroids)
                                       (* (sb-alien:unsigned 32))))
              (dst (sb-alien:sap-alien (sb-sys:vector-sap buf)
                                       (* (sb-alien:unsigned 32)))))
          (dotimes (i (length centroids))
            (setf (sb-alien:deref dst i) (sb-alien:deref src i)))))
      (write-sequence buf out))
    (let ((cbuf (make-array k :element-type '(unsigned-byte 32))))
      (loop for c below k do (setf (aref cbuf c) (aref counts c)))
      (write-sequence cbuf out))
    (write-sequence lists out)))

(defun run (&key (k *k*) (iters *iters*) (sample-size *sample*))
  (setf *k* k *iters* iters *sample* sample-size)
  (multiple-value-bind (vecs n) (load-vectors)
    (format t "loaded ~D vectors~%" n)
    (finish-output)
    (let* ((sample (random-sample n (min sample-size n)))
           (centroids (kmeans vecs sample k *dim* iters)))
      (format t "assigning all ~D rows to ~D centroids...~%" n k)
      (finish-output)
      (multiple-value-bind (counts lists) (assign-all vecs n centroids k *dim*)
        (save-ann centroids counts lists n k *dim*)
        (format t "done: ann.bin (k=~D, n=~D)~%" k n)))))
