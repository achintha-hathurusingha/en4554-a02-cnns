# EN4554 Assignment 2: CNNs

EN4554 Deep Learning for Vision, University of Moratuwa. Assignment 2 on CNNs (Ranga Rodrigo, 15 October 2026), due 22 October 2026.

One PyTorch Jupyter notebook on CIFAR-10 covering:

- **Task 1:** backpropagation from scratch, with gradient checks and vanishing gradients
- **Task 2:** SGD with momentum and Adam, implemented and compared
- **Task 3:** regularization (weight decay, dropout, flip+crop, cutout, mixup) and plain vs residual networks
- **Task 4 (optional):** the best CNN under 500k parameters and 20 epochs

All explanations and interpretations are in the notebook, which is exported to PDF for submission.

## Running it

The notebook runs top to bottom on a Google Colab T4 or a CUDA machine. CIFAR-10 is downloaded by `torchvision.datasets.CIFAR10` into `data/`, which isn't committed. Run histories are saved as JSON in `results/`, so the plotting cells can rebuild figures without retraining.

## Use of AI tools

Claude was used to help write and debug code and to draft explanations, as the assignment allows. The notebook's title cell states exactly how.
