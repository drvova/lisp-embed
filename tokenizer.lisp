;;;; tokenizer.lisp — universal tokenizer: WordPiece / BPE / Unigram.
;;;; Pipeline mirrors HuggingFace tokenizers:
;;;;   normalizer -> pre-tokenizer -> model -> post-processor
;;;;
;;;; References: llama.cpp llama-vocab.cpp (per-model pre-tokenizer regexes),
;;;; GPT-2 encoder.py (byte-level BPE + bytes_to_unicode), HF tokenizers
;;;; (normalizer/pre-tokenizer/model/post-processor decomposition).

(defpackage :tokenizer
  (:use :cl)
  (:export #:make-tokenizer
           #:make-tokenizer-from-json
           #:json-specials
           #:tk-encode
           #:tk-decode
           #:tk-vocab-size
           #:tk-unk-token
           #:tk-cls-token
           #:tk-sep-token
           #:tk-max-len
           #:tk-merges
           #:tk-scores
           #:*max-len*))
(in-package :tokenizer)

(defparameter *max-len* 512)

;;; ---- helpers ----

(defun whitespace-p (c) (member c '(#\Space #\Tab #\Newline #\Return #\Page)))

(defun control-p (c)
  (let ((cp (char-code c)))
    (or (< cp 32) (= cp 127) (and (>= cp 128) (< cp 160)))))

(defun punctuation-p (c)
  (let ((cp (char-code c)))
    (or (and (>= cp 33) (<= cp 47))
        (and (>= cp 58) (<= cp 64))
        (and (>= cp 91) (<= cp 96))
        (and (>= cp 123) (<= cp 126))
        (and (>= cp #x2000) (<= cp #x206F))
        (and (>= cp #x3000) (<= cp #x303F)))))

(defun cjk-p (c)
  (let ((cp (char-code c)))
    (or (and (>= cp #x4E00) (<= cp #x9FFF))
        (and (>= cp #x3400) (<= cp #x4DBF))
        (and (>= cp #xF900) (<= cp #xFAFF)))))

(defun strip-accents (s)
  "NFD decompose, drop combining marks, recompose NFC."
  (sb-unicode:normalize-string
   (with-output-to-string (out)
     (loop for c across (sb-unicode:normalize-string s :nfd)
           unless (plusp (sb-unicode:combining-class c))
             do (write-char c out)))
   :nfc))

(defun split-spaces (s)
  (let ((out nil) (start 0) (len (length s)))
    (loop for i of-type fixnum below len
          do (when (whitespace-p (char s i))
               (when (> i start) (push (subseq s start i) out))
               (setf start (1+ i))))
    (when (> len start) (push (subseq s start) out))
    (nreverse out)))

;;; ---- GPT-2 byte <-> unicode mapping (bytes_to_unicode) ----
;;; Maps every byte 0..255 to a printable unicode char so BPE operates on
;;; visible characters. Printable ASCII maps to itself; 0..32, 127..160, 173
;;; shift into 256+.

(defparameter *byte->unicode* (make-array 256 :element-type 'character))
(defparameter *unicode->byte* (make-hash-table))

(defun init-byte-maps ()
  (let ((n 0))
    (loop for b of-type fixnum below 256
          do (let ((cp (cond ((and (>= b 33) (<= b 126) (/= b 92)) b) ; printable ASCII minus backslash
                             ((and (>= b 161) (<= b 172)) b)
                             ((and (>= b 174) (<= b 255)) b)
                             (t (+ 256 (incf n) -1))))) ; shifted into 256+
               (setf (aref *byte->unicode* b) (code-char cp)
                     (gethash (code-char cp) *unicode->byte*) b)))))
(init-byte-maps)

(defun bytes-to-unicode-str (bytes)
  "Raw byte vector -> GPT-2 unicode-mapped string."
  (with-output-to-string (out)
    (loop for b across bytes do (write-char (aref *byte->unicode* b) out))))

(defun unicode-str-to-bytes (s)
  "GPT-2 unicode-mapped string -> raw byte vector."
  (let ((out (make-array (length s) :element-type '(unsigned-byte 8)
                                    :fill-pointer 0)))
    (loop for c across s
          do (vector-push (gethash c *unicode->byte*) out))
    out))

;;; ---- tokenizer struct ----

(defstruct (tokenizer (:constructor %make-tokenizer))
  kind        ; :wordpiece | :bpe | :unigram
  vocab       ; token string -> id hash
  inv-vocab   ; id -> token string vector
  merges      ; BPE: cons(a . b) -> rank hash
  scores      ; unigram: token -> log score hash
  max-len
  cls-token
  sep-token
  unk-token
  lowercase
  byte-level  ; BPE over GPT-2 unicode-mapped bytes
  pretok      ; :gpt2 | :llama3 pre-tokenizer regex
  bos-id      ; explicit BOS id (added tokens not in vocab.json)
  eos-id)     ; explicit EOS id

(defun make-tokenizer (kind &key vocab-path merges-path scores-path
                                 (lowercase t) (max-len 512)
                                 (cls-token "[CLS]") (sep-token "[SEP]")
                                 (unk-token "[UNK]")
                                 (byte-level nil) (pretok :gpt2) (bos-id nil) (eos-id nil))
  "Build a tokenizer.
:wordpiece — vocab.txt (one token per line, id = line number)
:bpe       — vocab.json (token->id) + merges.txt (ranked pairs)
:unigram   — scores-path: JSON {token: log-score}"
  (ecase kind
    (:wordpiece
     (let ((vocab (make-hash-table :test 'equal))
           (inv (make-array 4096 :element-type 'string :adjustable t
                                 :fill-pointer 0)))
       (with-open-file (in vocab-path :direction :input :external-format :utf-8)
         (loop for line = (read-line in nil nil)
               while line
               do (let ((tok (string-trim '(#\Return #\Space) line)))
                    (setf (gethash tok vocab) (fill-pointer inv))
                    (vector-push-extend tok inv))))
       (%make-tokenizer :kind :wordpiece :vocab vocab :inv-vocab inv
                        :lowercase lowercase :max-len max-len
                        :cls-token cls-token :sep-token sep-token
                        :unk-token unk-token
                        :bos-id bos-id :eos-id eos-id)))
    (:bpe
     (let ((vocab (make-hash-table :test 'equal))
           (inv (make-array 4096 :element-type 'string :adjustable t
                                 :fill-pointer 0))
           (merges (make-hash-table :test 'equal)))
       (with-open-file (in vocab-path :direction :input :external-format :utf-8)
         (let ((ht (json:parse (apply #'concatenate 'string
                                      (loop for line = (read-line in nil nil)
                                            while line
                                            collect line)))))
           (maphash (lambda (tok id)
                      (setf (gethash tok vocab) id)
                      (when (>= id (length inv))
                        (vector-push-extend tok inv)))
                    ht)))
       (with-open-file (in merges-path :direction :input :external-format :utf-8)
         (let ((rank 0)
               (first-line t))
           (loop for line = (read-line in nil nil)
                 while line
                 do (cond (first-line (setf first-line nil)) ; version line
                          ((and (plusp (length line))
                                (char/= (char line 0) #\#))
                           (let ((sp (position #\Space line)))
                             (when sp
                               (setf (gethash (cons (subseq line 0 sp)
                                                    (subseq line (1+ sp)))
                                              merges)
                                     rank)
                               (incf rank))))))))
       (%make-tokenizer :kind :bpe :vocab vocab :inv-vocab inv
                        :merges merges :lowercase lowercase
                        :max-len max-len :cls-token cls-token
                        :sep-token sep-token :unk-token unk-token
                        :byte-level byte-level :pretok pretok :bos-id bos-id :eos-id eos-id)))
    (:unigram
     (let ((vocab (make-hash-table :test 'equal))
           (inv (make-array 4096 :element-type 'string :adjustable t
                                 :fill-pointer 0))
           (scores (make-hash-table :test 'equal)))
       (with-open-file (in scores-path :direction :input :external-format :utf-8)
         (let ((ht (json:parse (apply #'concatenate 'string
                                      (loop for line = (read-line in nil nil)
                                            while line
                                            collect line)))))
           (maphash (lambda (tok score)
                      (setf (gethash tok vocab) (fill-pointer inv))
                      (vector-push-extend tok inv)
                      (setf (gethash tok scores) score))
                    ht)))
       (%make-tokenizer :kind :unigram :vocab vocab :inv-vocab inv
                        :scores scores :lowercase lowercase
                        :max-len max-len :cls-token cls-token
                        :sep-token sep-token :unk-token unk-token)))))

;;; ---- encode pipeline ----

(defun tk-encode (tk text)
  "Full pipeline: normalize -> pretokenize -> model ids -> [CLS]/[SEP] wrap.
Returns (values ids-vector attention-mask)."
  (let* ((normalized (if (tokenizer-lowercase tk)
                         (sb-unicode:lowercase text)
                         text))
         (words (ecase (tokenizer-kind tk)
                  (:wordpiece (pretokenize-wordpiece normalized))
                  (:bpe (pretokenize-bytelevel normalized (tokenizer-pretok tk)))
                  (:unigram (pretokenize-metaspace normalized))))
         (ids nil))
    (dolist (w words)
      (case (tokenizer-kind tk)
        (:wordpiece (setf ids (append ids (wordpiece-tokenize-word tk w))))
        (:bpe (setf ids (append ids
                                (bpe-merge tk
                                           (if (tokenizer-byte-level tk)
                                               (bytes-to-unicode-str
                                                (sb-ext:string-to-octets
                                                 w :external-format :utf-8))
                                               w)))))
        (:unigram (setf ids (append ids (unigram-viterbi tk w))))))
    (let* ((n (min (length ids) (- (tokenizer-max-len tk) 2)))
           (body (subseq ids 0 n))
           (vocab (tokenizer-vocab tk))
           (unk (gethash (tokenizer-unk-token tk) vocab))
           (cls (gethash (tokenizer-cls-token tk) vocab))
           (sep (gethash (tokenizer-sep-token tk) vocab))
           (body-ids (mapcar (lambda (tok) (or (gethash tok vocab) unk)) body))
           (bos (tokenizer-bos-id tk))
           (eos (tokenizer-eos-id tk))
           (total (+ (length body-ids) (if cls 1 0) (if sep 1 0)
                     (if bos 1 0) (if eos 1 0)))
           (idvec (make-array total :element-type 'fixnum))
           (mask (make-array total :element-type 'fixnum :initial-element 1))
           (pos 0))
      (when bos (setf (aref idvec pos) bos) (incf pos))
      (when cls (setf (aref idvec pos) cls) (incf pos))
      (dolist (id body-ids)
        (setf (aref idvec pos) id)
        (incf pos))
      (when sep (setf (aref idvec pos) sep) (incf pos))
      (when eos (setf (aref idvec pos) eos))
      (values idvec mask))))

(defun tk-decode (tk ids)
  "ids -> text via inverse vocab."
  (with-output-to-string (out)
    (loop for id across ids
          do (let ((tok (and (< id (length (tokenizer-inv-vocab tk)))
                             (aref (tokenizer-inv-vocab tk) id))))
               (when tok (write-string tok out))))))

(defun tk-vocab-size (tk) (length (tokenizer-inv-vocab tk)))

;;; ---- pre-tokenizers (called by tk-encode) ----

(defun pretokenize-wordpiece (s)
  "BERT style: whitespace split, then split punctuation to single chars."
  (let ((words (split-spaces s)) (out nil))
    (dolist (w words)
      (let ((start 0))
        (loop for i of-type fixnum below (length w)
              do (when (punctuation-p (char w i))
                   (when (> i start) (push (subseq w start i) out))
                   (push (subseq w i (1+ i)) out)
                   (setf start (1+ i))))
        (when (< start (length w))
          (push (subseq w start) out))))
    (nreverse out)))

(defun pretokenize-bytelevel (s &optional (mode :gpt2))
  "Byte-level BPE pre-split. :gpt2 = optional single space attaches to the
following run; :llama3 = optional single non-letter/digit char attaches to
letters, digits are single tokens (Qwen/Llama-3 style)."
  (if (eq mode :llama3)
      (pretokenize-llama3 s)
      (pretokenize-gpt2 s)))

(defun pretokenize-gpt2 (s)
  (let ((out nil) (i 0) (len (length s)))
    (labels ((alpha-at (j) (and (< j len) (alpha-char-p (char s j))))
             (digit-at (j) (and (< j len) (digit-char-p (char s j))))
             (punct-at (j) (and (< j len)
                                (let ((c (char s j)))
                                  (and (not (whitespace-p c))
                                       (not (alpha-char-p c))
                                       (not (digit-char-p c))))))
             (run (j pred)
               (let ((k j))
                 (loop while (funcall pred k) do (incf k))
                 k))
             (run-ws (j)
               (let ((k j))
                 (loop while (and (< k len) (whitespace-p (char s k))) do (incf k))
                 k)))
      (loop while (< i len) do
        (let ((c (char s i)))
          (cond
            ((and (char= c #\') (< (1+ i) len))
             (let* ((rest (subseq s (1+ i) (min len (+ i 3))))
                    (m (find-if (lambda (x) (and (<= (length x) (length rest))
                                                 (string= x rest :end2 (length x))))
                                '("s" "t" "m" "d" "re" "ve" "ll"))))
               (if m
                   (progn (push (subseq s i (+ i 1 (length m))) out)
                          (incf i (1+ (length m))))
                   (progn (push "'" out) (incf i)))))
            ((or (alpha-at i) (and (char= c #\Space) (alpha-at (1+ i))))
             (let* ((j (if (char= c #\Space) (1+ i) i))
                    (k (run j #'alpha-at)))
               (push (subseq s i k) out)
               (setf i k)))
            ((or (digit-at i) (and (char= c #\Space) (digit-at (1+ i))))
             (let* ((j (if (char= c #\Space) (1+ i) i))
                    (k (run j #'digit-at)))
               (push (subseq s i k) out)
               (setf i k)))
            ((or (punct-at i) (and (char= c #\Space) (punct-at (1+ i))))
             (let* ((j (if (char= c #\Space) (1+ i) i))
                    (k (run j #'punct-at)))
               (push (subseq s i k) out)
               (setf i k)))
            (t (let ((k (run-ws i)))
                 (push (subseq s i k) out)
                 (setf i k)))))))
    (nreverse out)))

(defun pretokenize-llama3 (s)
  (let ((out nil) (i 0) (len (length s)))
    (labels ((alpha-at (j) (and (< j len) (alpha-char-p (char s j))))
             (digit-at (j) (and (< j len) (digit-char-p (char s j))))
             (punct-at (j) (and (< j len)
                                (let ((c (char s j)))
                                  (and (not (whitespace-p c))
                                       (not (alpha-char-p c))
                                       (not (digit-char-p c))))))
             (run (j pred)
               (let ((k j))
                 (loop while (funcall pred k) do (incf k))
                 k))
             (run-ws (j)
               (let ((k j))
                 (loop while (and (< k len) (whitespace-p (char s k))) do (incf k))
                 k)))
      (loop while (< i len) do
        (let ((c (char s i)))
          (cond
            ((and (char= c #\') (< (1+ i) len))
             (let* ((rest (subseq s (1+ i) (min len (+ i 3))))
                    (m (find-if (lambda (x) (and (<= (length x) (length rest))
                                                 (string= x rest :end2 (length x))))
                                '("s" "t" "m" "d" "re" "ve" "ll"))))
               (if m
                   (progn (push (subseq s i (+ i 1 (length m))) out)
                          (incf i (1+ (length m))))
                   (progn (push "'" out) (incf i)))))
            ;; [^\r\n\p{L}\p{N}]?\p{L}+ : one leading non-letter/digit (space ok) attaches
            ((or (alpha-at i)
                 (and (not (member c '(#\Return #\Newline)))
                      (not (digit-at i)) (alpha-at (1+ i))))
             (let* ((j (if (alpha-at i) i (1+ i)))
                    (k (run j #'alpha-at)))
               (push (subseq s i k) out)
               (setf i k)))
            ;; \p{N} : single digit
            ((digit-at i) (push (subseq s i (1+ i)) out) (incf i))
            ;; ?[^\s\p{L}\p{N}]+ : optional space + punct run
            ((or (punct-at i) (and (char= c #\Space) (punct-at (1+ i))))
             (let* ((j (if (char= c #\Space) (1+ i) i))
                    (k (run j #'punct-at)))
               (push (subseq s i k) out)
               (setf i k)))
            (t (let ((k (run-ws i)))
                 (push (subseq s i k) out)
                 (setf i k)))))))
    (nreverse out)))

(defun pretokenize-metaspace (s)
  "SPM style: space becomes prefix of next word."
  (let ((words (split-spaces s)))
    (loop for w in words
          for first = t then nil
          collect (if first w (concatenate 'string " " w)))))

(defun pretokenize-metaspace (s)
  "SPM style: space becomes prefix of next word."
  (let ((words (split-spaces s)))
    (loop for w in words
          for first = t then nil
          collect (if first w (concatenate 'string " " w)))))

;;; ---- model tokenizers ----

(defun wordpiece-tokenize-word (tk word)
  "Greedy longest match for one pre-tokenized word."
  (if (> (length word) 100)
      (list (tokenizer-unk-token tk))
      (let ((pieces nil) (start 0) (len (length word)))
        (loop while (< start len)
              do (let ((end len) (found nil))
                   (loop while (> end start)
                         do (let ((sub (if (= start 0)
                                           (subseq word start end)
                                           (concatenate 'string "##" (subseq word start end)))))
                              (when (gethash sub (tokenizer-vocab tk))
                                (setf found sub)
                                (return))
                              (decf end)))
                   (if found
                       (progn (push found pieces)
                              (incf start (length found)))
                       (progn (push (tokenizer-unk-token tk) pieces)
                              (setf start len)))))
        (nreverse pieces))))

(defun bpe-merge (tk word)
  "Byte-level BPE merge by rank."
  (let ((parts (map 'list #'string word)))
    (loop
      (let ((best-rank most-positive-fixnum) (best-idx -1))
        (loop for i from 0 below (1- (length parts))
              do (let ((rank (gethash (cons (nth i parts) (nth (1+ i) parts))
                                      (tokenizer-merges tk))))
                   (when (and rank (< rank best-rank))
                     (setf best-rank rank best-idx i))))
        (if (= best-idx -1)
            (return parts)
            (setf parts
                  (append (subseq parts 0 best-idx)
                          (list (concatenate 'string
                                             (nth best-idx parts)
                                             (nth (1+ best-idx) parts)))
                          (subseq parts (+ best-idx 2)))))))))

(defun unigram-viterbi (tk s)
  "Viterbi over unigram vocab with log scores."
  (let* ((len (length s))
         (best (make-array (1+ len) :element-type 'double-float
                                    :initial-element most-negative-double-float))
         (back (make-array (1+ len) :element-type 'fixnum :initial-element -1)))
    (setf (aref best 0) 0.0d0)
    (loop for i from 0 below len
          do (loop for j from (+ i 1) to (min len (+ i 20))
                   do (let ((score (gethash (subseq s i j) (tokenizer-scores tk))))
                        (when score
                          (let ((cand (+ (aref best i) score)))
                            (when (> cand (aref best j))
                              (setf (aref best j) cand
                                    (aref back j) i)))))))
    (let ((ids nil) (j len))
      (loop while (> j 0)
            do (let ((i (aref back j)))
                 (push (gethash (subseq s i j) (tokenizer-vocab tk)) ids)
                 (setf j i)))
      ids)))

;;; ---- tokenizer.json (universal HF format) ----

(defun read-file-string (path)
  (with-open-file (in path :direction :input :external-format :utf-8)
    (let ((s (make-string (file-length in))))
      (subseq s 0 (read-sequence s in)))))

(defun json-specials (tj)
  "From a parsed tokenizer.json -> (values bos-id eos-id) via the
single-sequence TemplateProcessing. A SpecialToken before the Sequence is
BOS, after it is EOS."
  (let ((by-name (make-hash-table :test 'equal))
        (bos nil) (eos nil))
    (dolist (a (gethash "added_tokens" tj))
      (setf (gethash (gethash "content" a) by-name) (gethash "id" a)))
    (let* ((pp (gethash "post_processor" tj))
           (pp (and (hash-table-p pp) pp))
           (ptype (and pp (gethash "type" pp))))
      (cond
        ((and ptype (member ptype '("RobertaProcessing" "BertProcessing")
                            :test #'string=))
         (let ((cls (gethash "cls" pp))
               (sep (gethash "sep" pp)))
           (setf bos (and cls (second cls))
                 eos (and sep (second sep)))))
        (t
         (let* ((procs (and pp (gethash "processors" pp)))
                (procs (and (listp procs) procs))
                (tp (and procs
                         (find-if (lambda (p) (string= (gethash "type" p) "TemplateProcessing"))
                                  procs))))
           (when tp
             (let ((seen-seq nil))
               (dolist (item (gethash "single" tp))
                 (let ((tok (gethash "SpecialToken" item)))
                   (when tok
                     (let ((id (gethash (gethash "id" tok) by-name)))
                       (if seen-seq (setf eos id) (setf bos id)))))
                 (when (gethash "Sequence" item)
                   (setf seen-seq t)))))))))
    (values bos eos)))

(defun pretok-mode (tj)
  "Detect the pre-tokenizer regex family from tokenizer.json."
  (let* ((pre (gethash "pre_tokenizer" tj))
         (pre (and (hash-table-p pre) pre))
         (procs (and pre (gethash "pretokenizers" pre)))
         (pat nil))
    (dolist (p (or procs (list pre)))
      (let ((split (gethash "pattern" p)))
        (when split
          (let ((rx (gethash "Regex" split)))
            (when rx (setf pat rx))))))
    (if (and pat (search "\\r\\n" pat))
        :llama3
        :gpt2)))

(defun make-tokenizer-from-json (path)
  "Build a tokenizer from a HuggingFace tokenizer.json (WordPiece or BPE).
Specials come from the post-processor; the pre-split regex family is detected."
  (let* ((tj (json:parse (read-file-string path)))
         (model (gethash "model" tj))
         (mtype (gethash "type" model))
         (vocab (gethash "vocab" model))
         (norm (gethash "normalizer" tj))
         (lowercase (and (hash-table-p norm) (gethash "lowercase" norm)))
         (maxid 0))
    (maphash (lambda (tok id) (declare (ignore tok)) (setf maxid (max maxid id))) vocab)
    (let ((inv (make-array (1+ maxid) :element-type 'string :initial-element "")))
      (maphash (lambda (tok id) (setf (aref inv id) tok)) vocab)
      (multiple-value-bind (bos eos) (json-specials tj)
        (cond
          ((string= mtype "WordPiece")
           (%make-tokenizer :kind :wordpiece :vocab vocab :inv-vocab inv
                            :lowercase (and lowercase t) :max-len 512
                            :cls-token (if bos "" "[CLS]")
                            :sep-token (if eos "" "[SEP]") :unk-token "[UNK]"
                            :bos-id bos :eos-id eos))
          ((string= mtype "BPE")
           (let ((merges (make-hash-table :test 'equal))
                 (rank 0))
             (dolist (m (gethash "merges" model))
               (let ((pair (if (consp m)
                               (cons (first m) (second m))
                               (let ((sp (position #\Space m)))
                                 (and sp (cons (subseq m 0 sp) (subseq m (1+ sp))))))))
                 (when pair
                   (setf (gethash pair merges) rank)
                   (incf rank))))
             (%make-tokenizer :kind :bpe :vocab vocab :inv-vocab inv :merges merges
                              :lowercase nil :max-len 512
                              :cls-token "" :sep-token "" :unk-token "<unk>"
                              :pretok (pretok-mode tj) :byte-level t
                              :bos-id bos :eos-id eos)))
          (t (error "unsupported tokenizer.json model type ~A" mtype)))))))
