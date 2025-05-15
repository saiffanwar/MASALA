for starting_k in 4 8 12 16; do
    for sparsity in 0.02 0.05 0.1 0.2; do
        python chilli_runner.py --starting_k $starting_k --sparsity $sparsity --masala
    done
done
