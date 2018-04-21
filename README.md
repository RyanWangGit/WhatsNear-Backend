# WhatsNear-Backend [![Build Status](https://www.travis-ci.org/RyanWangGit/WhatsNear-Backend.svg?branch=master)](https://www.travis-ci.org/RyanWangGit/WhatsNear-Backend) [![codecov](https://codecov.io/gh/RyanWangGit/WhatsNear-Backend/branch/master/graph/badge.svg)](https://codecov.io/gh/RyanWangGit/WhatsNear-Backend)

This Project Serves as the Backend for [WhatsNear](https://github.com/RyanWangGit/WhatsNear).

The [WhastNear](https://github.com/RyanWangGit/WhatsNear) project uses Github Pages to present itself to the public, which will create data requests to this backend program.

# Problem Formulation
Refer to [WhatNear-Problem Formulation](https://github.com/RyanWangGit/WhatsNear/blob/source/README.md#problem-formulation)

## Usage

Simply run `python __main__.py [args]` to learn from the train data and starts the HTTP server, the available arguments are listed as follows:

| Argument                  | Description                           | 
| ------------------------- | ------------------------------------- |
| -h, --help                | Show help message and exit.           |
| -p PORT, --port PORT      | The port to listen on.                |
|-s SQLITE, --sqlite SQLITE |The SQLite3 database to read from.     |
|-t TRAIN, --train TRAIN    |The training matrix file to read from. |
|-i IP, --ip IP             |The ip to bind on.                     |
|-m MODEL, --model MODEL    |The trained model to read from.        |

## Third-Party Libraries
This very project uses [Tornado](https://github.com/tornadoweb/tornado) as the HTTP Server and uses [Keras](https://github.com/fchollet/keras) as the neural network backend to train the models.

## References
[1] Burges C, Shaked T, Renshaw E, et al. Learning to rank using gradient descent[C]//Proceedings of the 22nd international conference on Machine learning. ACM, 2005: 89-96.

## License
[MIT](https://github.com/RyanWangGit/WhatsNear-Backend/blob/master/LICENSE).

