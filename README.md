# My personal code library
This repository contains various small projects, scripts, and other miscellanea that are too small to deserve their own repository.

## Contents
### theano-musings
Examples of GPU neural net training code using
[Theano](http://deeplearning.net/software/theano/) and
[Lasagne](https://lasagne.readthedocs.io/en/latest/).

### Presto period divider script
A script to find the best arrangement of periods to claim the cost of transit trips under the
[Federal Public Transit Tax Credit](https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/deductions-credits-expenses/line-364-public-transit-amount.html)
for users of the [TTC's Presto card](https://www.prestocard.ca/en/about/tax-credit).

The program reads the generated CSV file directly, and runs a dynamic programming optimization algorithm over the data.
