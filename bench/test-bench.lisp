;;;; test-bench.lisp — embedding-model benchmark on the cybersec candidate pool.
;;;; pool.jsonl: 10 queries x top-20 from the live index (bge-small).
;;;; Each model re-ranks the top-10 pool per query; keyword labels score P@5.

(load "D:/cybersec-kb/json.fasl")
(load "D:/cybersec-kb/st.fasl")
(load "D:/cybersec-kb/tokenizer.fasl")
(load "D:/cybersec-kb/llama.fasl")
(load "D:/cybersec-kb/bert.fasl")

(defparameter *queries*
  '("how to harden an SSH server"
    "detecting SQL injection in a web application"
    "kerberoasting attack detection"
    "buffer overflow exploit development"
    "Windows registry persistence techniques"
    "wireshark packet capture analysis"
    "XSS payload filter bypass"
    "privilege escalation on Linux"
    "malware sandbox analysis"
    "password cracking with hashcat"))

(defun contains (hay needle)
  (not (null (search needle hay))))

(defun relevant-p (qi text)
  "Keyword relevance label per query (lowercased substring checks)."
  (let ((s (string-downcase text)))
    (case qi
      (0 (and (contains s "ssh")
              (or (contains s "harden") (contains s "hardening"))))
      (1 (and (contains s "sql") (contains s "injection")))
      (2 (contains s "kerberoast"))
      (3 (and (contains s "buffer") (contains s "overflow")))
      (4 (and (contains s "registry")
              (or (contains s "persist") (contains s "run key")
                  (contains s "autorun"))))
      (5 (contains s "wireshark"))
      (6 (or (contains s "xss") (contains s "cross-site scripting")))
      (7 (and (contains s "privilege") (contains s "escalat")))
      (8 (contains s "sandbox"))
      (9 (contains s "hashcat"))
      (t nil))))

(defun read-pool (path)
  "List of (q rank score text)."
  (with-open-file (in path :direction :input :external-format :utf-8)
    (loop for line = (read-line in nil nil)
          while line
          for obj = (json:parse line)
          collect (list (gethash "q" obj) (gethash "rank" obj)
                        (gethash "score" obj) (gethash "text" obj)))))

(defun short-text (row)
  (let ((txt (fourth row)))
    (subseq txt 0 (min 200 (length txt)))))

(defun dot (a b)
  (let ((s 0.0))
    (loop for i below (length a) do (incf s (* (aref a i) (aref b i))))
    s))

(defun bench-model (label path &key query-prefix doc-prefix)
  (handler-case
      (let ((bert:*query-prefix* (or query-prefix ""))
            (pool (read-pool "D:/cybersec-kb/pool.jsonl")))
        (bert:load-model path)
        (format *error-output* "[~A] model loaded~%" label) (finish-output *error-output*)
        ;; query latency: 3 timed embeds, take the min
        (bert:embed (first *queries*))
        (format *error-output* "[~A] warmup done~%" label) (finish-output *error-output*)
        (let ((lat (reduce #'min
                           (loop repeat 3
                                 collect (let ((t0 (get-internal-real-time)))
                                           (bert:embed (nth 1 *queries*))
                                           (/ (- (get-internal-real-time) t0)
                                              (/ internal-time-units-per-second 1000.0)))))))
          ;; embed unique pool texts once
          (let ((cache (make-hash-table :test 'equal))
                (qvecs (mapcar (lambda (q) (bert:embed q)) *queries*)))
            (let ((cnt 0))
              (dolist (row pool)
                (when (< (second row) 5)
                  (let ((txt (short-text row)))
                    (unless (gethash txt cache)
                      (incf cnt)
                      (setf (gethash txt cache)
                            (bert:embed (concatenate 'string (or doc-prefix "") txt)))))))
              (format *error-output* "[~A] embedded ~D pool texts~%" label cnt)
              (finish-output *error-output*))
            ;; score
            (let ((p5 0.0) (p10 0.0) (top1 0.0) (nq 0) (dim 0) (labeled 0))
              (loop for qi below (length *queries*) do
                (let* ((rows (remove-if-not (lambda (r) (and (= (first r) qi)
                                                             (< (second r) 5)))
                                            pool))
                       (scored (mapcar (lambda (r)
                                         (cons (dot (nth qi qvecs) (gethash (short-text r) cache))
                                               r))
                                       rows))
                       (sorted (sort scored #'> :key #'car))
                       (rels (mapcar (lambda (r) (relevant-p qi (short-text (cdr r)))) sorted)))
                  (incf nq)
                  (incf labeled (count t rels))
                  (incf p5 (/ (count t (subseq rels 0 (min 5 (length rels)))) 5.0))
                  (incf p10 (/ (count t (subseq rels 0 (min 10 (length rels)))) 10.0))
                  (incf top1 (if (first rels) 1.0 0.0))
                  (setf dim (length (nth qi qvecs)))))
              (format t "~A|~D|~,3F|~,3F|~,3F|~,1F|~D~%"
                      label dim (/ p5 nq) (/ p10 nq) (/ top1 nq) lat labeled)
              (finish-output)))))
    (error (e) (format t "~A|ERR|~A~%" label e) (finish-output))))

(defparameter *models*
  '(("bge-small-en-v1.5"
     "C:/Users/bigmo/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model.safetensors"
     nil nil)
    ("all-MiniLM-L6-v2"
     "C:/Users/bigmo/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors"
     nil nil)
    ("e5-small-v2"
     "C:/Users/bigmo/.cache/huggingface/hub/models--intfloat--e5-small-v2/snapshots/ffb93f3bd4047442299a41ebb6fa998a38507c52/model.safetensors"
     "query: " "passage: ")
    ("gte-small"
     "C:/Users/bigmo/.cache/huggingface/hub/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd/model.safetensors"
     nil nil)
    ("arctic-embed-s"
     "C:/Users/bigmo/.cache/huggingface/hub/models--Snowflake--snowflake-arctic-embed-s/snapshots/e596f507467533e48a2e17c007f0e1dacc837b33/model.safetensors"
     nil nil)
    ("bge-base-en-v1.5"
     "C:/Users/bigmo/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/model.safetensors"
     nil nil)
    ("all-mpnet-base-v2"
     "C:/Users/bigmo/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/model.safetensors"
     nil nil)
    ("e5-base-v2"
     "C:/Users/bigmo/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/model.safetensors"
     "query: " "passage: ")
    ("gte-base"
     "C:/Users/bigmo/.cache/huggingface/hub/models--thenlper--gte-base/snapshots/c078288308d8dee004ab72c6191778064285ec0c/model.safetensors"
     nil nil)
    ("arctic-embed-m"
     "C:/Users/bigmo/.cache/huggingface/hub/models--Snowflake--snowflake-arctic-embed-m/snapshots/fc74610d18462d218e312aa986ec5c8a75a98152/model.safetensors"
     nil nil)
    ("bge-large-en-v1.5"
     "C:/Users/bigmo/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09/model.safetensors"
     nil nil)
    ("mxbai-embed-large-v1"
     "C:/Users/bigmo/.cache/huggingface/hub/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/b33106f585b9ce46904ad7443a3b52b7a63e231c/model.safetensors"
     "Represent this sentence for searching relevant passages: " nil)
    ("Qwen3-Embedding-0.6B"
     "C:/Users/bigmo/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/model.safetensors"
     nil nil)))

(dolist (m *models*)
  (bench-model (first m) (second m) :query-prefix (third m) :doc-prefix (fourth m)))
