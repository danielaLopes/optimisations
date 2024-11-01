from memory_profiler import profile
import numpy as np
import time


class DatasetHandler:
    def __init__(self, filename):
        self.filename = filename
        self._data = None  # Data isn't loaded initially

    @property
    @profile
    def data(self):
        if self._data is None:  # Load only when first accessed
            print("Loading data...")
            # Simulate loading a large dataset
            time.sleep(1)  # Simulate I/O operation
            self._data = np.random.rand(int(1e7))
        return self._data

    def get_summary(self):
        # Data is loaded only if this method is called
        return {
            'mean': self.data.mean(),
            'std': self.data.std()
        }


# Let's see it in action
@profile
def demo_lazy_loading():
    # Creating the object is very light
    print("Creating object...")
    handler = DatasetHandler('large_dataset.csv')
    print("Object created, memory footprint is small")

    # Do some work that doesn't need the data
    time.sleep(1)
    print("Did some other work without loading data")

    # Now actually use the data
    print("Computing summary statistics...")
    stats = handler.get_summary()
    print(f"Stats computed: mean={stats['mean']:.2f}, std={stats['std']:.2f}")


if __name__ == '__main__':
    handler = DatasetHandler('large_dataset.csv')
    print("No memory used yet!")

    # Memory is only allocated when we actually access the data
    stats = handler.get_summary()