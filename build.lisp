;;;; build.lisp — compile and load the engine in dependency order.
;;;;   sbcl --load build.lisp
(let ((dir *load-pathname*))
  (dolist (f '("config" "json" "st" "tokenizer" "llama" "bert" "gemma" "model"
               "ann" "embed" "extract" "chunk" "kb" "serve"))
    (let ((src (merge-pathnames (concatenate 'string f ".lisp") dir)))
      (compile-file src)
      (load (merge-pathnames (concatenate 'string f ".fasl") dir)))))
(format t "built.~%")
