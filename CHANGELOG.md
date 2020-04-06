Unreleased
==========

0.1.4 (2020-4-6)
==================
- Depreciated getters; please access the data directly from now on.
- Removed unnecessary iteration benchmarks.

0.1.3 (2020-1-31)
==================
- Added the option to use iterators to access data.
- Added several benchmarks (run using "cargo bench")
- Removed "/" from file paths to progress towards Windows support.
  - This is a breaking change; please see the diff for the perceptron.rs example.

0.1.2 (2020-1-19)
==================
- Fixed the example link in the README.md.


0.1.1 (2020-1-19)
==================

- Significant performance improvement (now stores images as array rather than vector).
  - This is a breaking change, as your functions will now recieve an `Array` type as opposed to a 
  `Vec`.
- Removed redundant debug output.

0.1.0 (2020-1-18)
==================

First release, version 0.1.0.
