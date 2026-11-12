;;;; serve.lisp — pure-Common-Lisp MCP stdio server for the cybersec KB.
;;;; JSON-RPC 2.0 over stdin/stdout (json.lisp); index + inference all Lisp.
;;;;
;;;; Start:  sbcl --dynamic-space-size 8192 --load serve.lisp

(load (merge-pathnames "json.fasl" *load-pathname*))
(load (merge-pathnames "st.fasl" *load-pathname*))
(load (merge-pathnames "tokenizer.fasl" *load-pathname*))
(load (merge-pathnames "llama.fasl" *load-pathname*))
(load (merge-pathnames "bert.fasl" *load-pathname*))
(load (merge-pathnames "gemma.fasl" *load-pathname*))
(load (merge-pathnames "model.fasl" *load-pathname*))
(load (merge-pathnames "kb.fasl" *load-pathname*))
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :kb)
    (make-package :kb :use '(:cl))))
(in-package :kb)

(defun jobj (&rest pairs)
  "Build a hash-table object from alternating key/value args."
  (let ((h (make-hash-table :test 'equal)))
    (loop for (k v) on pairs by #'cddr
          do (setf (gethash k h) v))
    h))

(defun tool-schemas ()
  "MCP tools/list result."
  (list
   (jobj "name" "kb_search"
         "description" "Semantic search over the cybersecurity knowledge base (3.2M chunks, 32 domains)."
         "inputSchema" (jobj "type" "object"
                             "properties" (jobj "query" (jobj "type" "string")
                                                "k" (jobj "type" "integer")
                                                "domain" (jobj "type" "string"))
                             "required" (list "query")))
   (jobj "name" "kb_inject"
         "description" "Ready-to-paste RAG context block for a query, char-budgeted."
         "inputSchema" (jobj "type" "object"
                             "properties" (jobj "query" (jobj "type" "string")
                                                "k" (jobj "type" "integer")
                                                "budget" (jobj "type" "integer")
                                                "domain" (jobj "type" "string"))
                             "required" (list "query")))
   (jobj "name" "kb_stats"
         "description" "Chunk counts per domain."
         "inputSchema" (jobj "type" "object" "properties" (jobj)))))

(defun tool-result (text)
  (jobj "content" (list (jobj "type" "text" "text" text))
        "isError" nil))

(defun handle-tool-call (name args)
  (cond ((string= name "kb_search")
         (let* ((query (or (gethash "query" args) ""))
                (k (or (gethash "k" args) 8))
                (domain (gethash "domain" args))
                (hits (kb-search query k)))
           (when domain
             (setf hits (remove-if-not
                         (lambda (h) (string= (getf h :domain) domain))
                         hits)))
           (tool-result
            (with-output-to-string (out)
              (loop for h in hits
                    do (format out "[~A | ~A]~%~A~%~%"
                               (getf h :source) (getf h :domain)
                               (getf h :text)))))))
        ((string= name "kb_inject")
         (let ((query (or (gethash "query" args) ""))
               (k (or (gethash "k" args) 8))
               (budget (or (gethash "budget" args) 12000)))
           (tool-result (kb-inject query k budget))))
        ((string= name "kb_stats")
         (tool-result
          (with-output-to-string (out)
            (loop for (d . c) in (kb-stats)
                  do (format out "~A: ~:d~%" d c)))))
(t (tool-result "unknown tool"))))

(defun rpc-error (id code message)
  (jobj "jsonrpc" "2.0" "id" id
        "error" (jobj "code" code "message" message)))

(defun handle-rpc (msg)
  "Handle one parsed JSON-RPC message; returns the response string or NIL."
  (let ((method (gethash "method" msg))
        (id (gethash "id" msg))
        (params (gethash "params" msg)))
    (cond ((string= method "initialize")
           (json:write-json
            (jobj "jsonrpc" "2.0" "id" id
                  "result" (jobj "protocolVersion"
                                 (or (and params (gethash "protocolVersion" params))
                                     "2024-11-05")
                                 "capabilities" (jobj "tools" (jobj "listChanged" nil))
                                 "serverInfo" (jobj "name" "cybersec-kb" "version" "1.0")))))
          ((string= method "notifications/initialized") nil)
          ((string= method "tools/list")
           (json:write-json
            (jobj "jsonrpc" "2.0" "id" id
                  "result" (jobj "tools" (tool-schemas)))))
          ((string= method "tools/call")
           (let ((name (gethash "name" params))
                 (args (gethash "arguments" params)))
             (json:write-json
              (jobj "jsonrpc" "2.0" "id" id
                    "result" (handle-tool-call name args)))))
          (t (and id (rpc-error id -32601 "method not found"))))))

(defun serve ()
  (loop
    (let ((line (read-line *standard-input* nil nil)))
      (when (null line) (return))
      (handler-case
          (let ((resp (progn (format *error-output* "RPC: ~A~%" (subseq line 0 (min 60 (length line)))) (handle-rpc (json:parse line)))))
            (when resp
              (write-line resp *standard-output*)
              (finish-output *standard-output*)))
        (error (e)
          (format *error-output* "serve error: ~A~%" e))))))

(when (member :serve *features*)
  (kb:load-index)
  (kb:start-encoder)
  (serve))
