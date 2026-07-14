# Static Tree Compression Benchmark

This mini project downloads the UCI Mushroom poisonous/edible dataset, trains
either a decision tree or a random forest, converts the learned Python object
graph into a compact static array representation, then reports memory and
prediction-time savings.

The compact representation follows the practical idea behind space-efficient
static tree structures: store fixed topology and node payloads in contiguous
typed arrays instead of pointer-heavy Python objects.
