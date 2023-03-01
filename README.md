
# PrimeGPT

A large language model for finding patterns in prime numbers. 


## Acknowledgements

 - [Inspiration for PrimeGPT is Prime95](https://www.mersenne.org/download/)

## Appendix

You can learn about prime numbers here, a curated list of prime number projects.

https://github.com/PlummersSoftwareLLC/Primes


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

## Demo

Run training.py to get started.


## FAQ

Can you really predict prime numbers with machine learning?

Yes, I used kernel regression to get 0.99 accuracy using 3-fold cross validation first 1000 prime numbers. A success here was defined as being "close enough" to a prime number with a threshold < 1.

#### What methods are used in this project

Kernel regression, Bayesian optimization (an attempt to calculate Miller's constant) and a large language model (based on a character level GPT) are used to forecast prime numbers. Several additional algorithms are provided in the algml library.



## Roadmap

- Add distributed search functionality to search for primes in larger ranges.



## Contributing

Contributions are always welcome! Submit a PR.


