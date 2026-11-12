;;;; start.lisp — boot the search runtime (index + encoder + REPL banner).
;;;;   sbcl --dynamic-space-size 8192 --load start.lisp
(dolist (f '("config" "json" "st" "tokenizer" "llama" "bert" "gemma" "model" "ann" "kb"))
  (load (merge-pathnames (concatenate 'string f ".fasl") *load-pathname*)))
(kb:load-index)
(kb:start-encoder)
(kb:repl-banner)
