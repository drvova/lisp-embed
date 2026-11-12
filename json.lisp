;;;; json.lisp — minimal JSON parser/writer (RFC 8259 subset sufficient for
;;;; safetensors headers and the MCP stdio protocol).
;;;;
;;;; The parser works on a string + index (no stream read-char per character),
;;;; which is several times faster than a character-stream reader for large
;;;; corpora.

(defpackage :json
  (:use :cl)
  (:export #:parse #:write-json))
(in-package :json)

(defstruct (jp (:constructor make-jp (str)))
  (str "" :type string)
  (pos 0 :type fixnum)
  (len 0 :type fixnum))

(defun jp-new (str)
  (let ((p (make-jp str)))
    (setf (jp-len p) (length str))
    p))

(declaim (inline jp-peek jp-next jp-eof))
(defun jp-peek (p)
  (let ((i (jp-pos p)))
    (when (< i (jp-len p)) (char (jp-str p) i))))
(defun jp-next (p)
  (let ((i (jp-pos p)))
    (prog1 (char (jp-str p) i) (setf (jp-pos p) (1+ i)))))
(defun jp-eof (p) (>= (jp-pos p) (jp-len p)))

(defun skip-ws (p)
  (loop for c = (jp-peek p)
        while (and c (member c '(#\Space #\Tab #\Newline #\Return)))
        do (jp-next p)))

(defun read-n-chars (p n)
  (let ((start (jp-pos p)))
    (setf (jp-pos p) (min (jp-len p) (+ start n)))
    (subseq (jp-str p) start (jp-pos p))))

(defparameter *escape-chars*
  '((#\" . #\") (#\\ . #\\) (#\/ . #\/) (#\b . #\Backspace)
    (#\f . #\Page) (#\n . #\Newline) (#\r . #\Return) (#\t . #\Tab)))

(defun escape-char (e out)
  "Push the decoded escape character E onto OUT; returns T if handled."
  (let ((pair (assoc e *escape-chars*)))
    (if pair
        (progn (vector-push-extend (cdr pair) out) t)
        nil)))

(defun parse-string (p)
  (jp-next p) ; opening quote
  (let ((out (make-array 32 :element-type 'character :fill-pointer 0 :adjustable t)))
    (loop
      (let ((c (jp-next p)))
        (cond ((char= c #\") (return out))
              ((char= c #\\)
               (let ((e (jp-next p)))
                 (if (char= e #\u)
                     (let ((cp (parse-integer (read-n-chars p 4) :radix 16)))
                       (if (<= 55296 cp 56319)
                           ;; high surrogate: combine with the following \uXXXX
                           (progn
                             (jp-next p) ; backslash
                             (jp-next p) ; u
                             (let ((lo (parse-integer (read-n-chars p 4) :radix 16)))
                               (vector-push-extend
                                (code-char (+ 65536 (ash (- cp 55296) 10) (- lo 56320)))
                                out)))
                           ;; normal BMP char (incl. variation selectors)
                           (vector-push-extend (code-char cp) out)))
                     (if (escape-char e out)
                         nil
                         (error "bad escape ~A" e)))))
              (t (vector-push-extend c out)))))))

(defun parse-number (p)
  (let ((start (jp-pos p)))
    (loop for c = (jp-peek p)
          while (and c (or (digit-char-p c) (member c '(#\- #\+ #\. #\e #\E))))
          do (jp-next p))
    (let ((str (subseq (jp-str p) start (jp-pos p))))
      (if (or (find #\. str) (find #\e str) (find #\E str))
          (let ((mantissa 0d0) (sign 1d0) (i 0) (len (length str)))
            (when (char= (char str 0) #\-) (setf sign -1d0 i 1))
            (loop while (< i len)
                  for c = (char str i)
                  while (digit-char-p c)
                  do (setf mantissa (+ (* mantissa 10) (digit-char-p c)))
                     (incf i))
            (when (and (< i len) (char= (char str i) #\.))
              (incf i)
              (let ((scale 0.1d0))
                (loop while (< i len)
                      for c = (char str i)
                      while (digit-char-p c)
                      do (setf mantissa (+ mantissa (* scale (digit-char-p c))))
                         (setf scale (/ scale 10d0))
                         (incf i))))
            (when (and (< i len) (or (char= (char str i) #\e) (char= (char str i) #\E)))
              (incf i)
              (let ((esign 1) (e 0))
                (when (and (< i len) (char= (char str i) #\-)) (setf esign -1) (incf i))
                (when (and (< i len) (char= (char str i) #\+)) (incf i))
                (loop while (< i len)
                      for c = (char str i)
                      while (digit-char-p c)
                      do (setf e (+ (* e 10) (digit-char-p c)))
                         (incf i))
                (setf mantissa (* mantissa (expt 10d0 (* esign e))))))
            (* sign mantissa))
          (parse-integer str)))))

(defun read-word (p word)
  (loop for expect across word
        do (assert (char= (jp-next p) expect))))

(defun parse-object (p)
  (jp-next p) ; {
  (let ((ht (make-hash-table :test 'equal)))
    (skip-ws p)
    (when (char= (jp-peek p) #\})
      (jp-next p) (return-from parse-object ht))
    (loop
      (skip-ws p)
      (let ((key (parse-string p)))
        (skip-ws p)
        (assert (char= (jp-next p) #\:))
        (setf (gethash key ht) (parse-value p)))
      (skip-ws p)
      (let ((c (jp-next p)))
        (cond ((char= c #\,))
              ((char= c #\}) (return ht))
              (t (error "bad object separator ~A" c)))))))

(defun parse-array (p)
  (jp-next p) ; [
  (let ((out nil))
    (skip-ws p)
    (when (char= (jp-peek p) #\])
      (jp-next p) (return-from parse-array (nreverse out)))
    (loop
      (push (parse-value p) out)
      (skip-ws p)
      (let ((c (jp-next p)))
        (cond ((char= c #\,))
              ((char= c #\]) (return (nreverse out)))
              (t (error "bad array separator ~A" c)))))))

(defun parse-value (p)
  (skip-ws p)
  (let ((c (jp-peek p)))
    (cond ((char= c #\{) (parse-object p))
          ((char= c #\[) (parse-array p))
          ((char= c #\") (parse-string p))
          ((char= c #\t) (read-word p "true") t)
          ((char= c #\f) (read-word p "false") nil)
          ((char= c #\n) (read-word p "null") :null)
          (t (parse-number p)))))

(defun parse (string)
  (parse-value (jp-new string)))

;;; ---- writer ----

(defun write-json-escaped-string (str out)
  (format out "\"")
  (loop for c across str
        do (cond ((char= c #\") (format out "\\\""))
                 ((char= c #\\) (format out "\\\\"))
                 ((char= c #\Newline) (format out "\\n"))
                 ((char= c #\Return) (format out "\\r"))
                 ((char= c #\Tab) (format out "\\t"))
                 ((< (char-code c) 32)
                  (format out "\\u~4,'0x" (char-code c)))
                 (t (write-char c out))))
  (format out "\""))

(defun write-json-key (k out)
  (write-json-escaped-string (string k) out))

(defun write-json (value)
  (with-output-to-string (out)
    (typecase value
      (hash-table
       (format out "{")
       (let ((first t))
         (maphash (lambda (k v)
                    (unless first (format out ","))
                    (setf first nil)
                    (write-json-key k out)
                    (format out ":")
                    (format out "~A" (write-json v)))
                  value))
       (format out "}"))
      (null
       (format out "[]"))
      (string
       (write-json-escaped-string value out))
      (cons
       (if (every #'consp value)
           (progn
             (format out "{")
             (let ((first t))
               (dolist (pair value)
                 (unless first (format out ","))
                 (setf first nil)
                 (write-json-key (car pair) out)
                 (format out ":")
                 (format out "~A" (write-json (cdr pair))))
               (format out "}")))
           (progn
             (format out "[")
             (let ((first t))
               (dolist (v value)
                 (unless first (format out ","))
                 (setf first nil)
                 (format out "~A" (write-json v)))
               (format out "]")))))
      (vector
       (format out "[")
       (loop for v across value
             for first = t then nil
             unless first do (format out ",")
             do (format out "~A" (write-json v)))
       (format out "]"))
      (symbol (cond ((eq value t) (format out "true"))
                    ((eq value nil) (format out "false"))
                    ((eq value :null) (format out "null"))
                    (t (write-json-escaped-string (string value) out))))
      (integer (format out "~D" value))
      (float (format out "~F" value))
      (t (write-json-escaped-string (princ-to-string value) out)))))
