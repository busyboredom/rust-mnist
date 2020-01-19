Rust-mnist is an MNIST dataset parser written in rust. It is simple and lightweight, not 
unreasonably slow and features some helpful error messages to get you started.

Using rust-mnist
----------------

See [examples/perceptron.rs](../blob/master/examples/perceptron.rs) for a rudamentary 
demonstration.


You will need to download and extract the dataset from http://yann.lecun.com/exdb/mnist/ 
before use.

You may also want to add rust-mnist to your Cargo.toml, so that Cargo can manage it as a 
dependency for you:

```TOML
[dependencies]
rust-mnist = "0.1"
```

Finally, you may want to use some logging implementation if you'd like to see `info` or `debug` 
information from rust-mnist. I'd recommend [fern](https://docs.rs/fern/0.5.9/fern/).
