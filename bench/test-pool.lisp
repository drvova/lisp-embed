(load "D:/cybersec-kb/json.fasl")
(load "D:/cybersec-kb/st.fasl")
(load "D:/cybersec-kb/tokenizer.fasl")
(load "D:/cybersec-kb/llama.fasl")
(load "D:/cybersec-kb/bert.fasl")
(load "D:/cybersec-kb/kb.lisp")

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

(cybersec-kb:load-index)
(bert:load-model "C:/Users/bigmo/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model.safetensors")
(setf cybersec-kb::*encoder* t)
(with-open-file (out "D:/cybersec-kb/pool.jsonl" :direction :output :if-exists :supersede)
  (loop for q in *queries*
        for qi from 0
        do (let ((hits (cybersec-kb:kb-search q 20)))
             (loop for h in hits
                   for r from 0
                   do (format out "~A~%"
                              (json:write-json
                               (list (cons "q" qi) (cons "rank" r)
                                     (cons "score" (getf h :score))
                                     (cons "text" (subseq (getf h :text) 0 (min 600 (length (getf h :text)))))))))))
  (format t "pool written~%"))
