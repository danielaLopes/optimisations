# Python version - creates many copies in memory
def matrix_multiply_python(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])

    # Creates new memory allocation
    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            # Creates temporary allocations for each sum
            result[i][j] = sum(a[i][k] * b[k][j] for k in range(cols_a))
    return result


# Mojo version - optimized memory usage
struct
Matrix:
var
data: DTypePointer[DType.float64]
var
rows: Int
var
cols: Int

fn
__init__(inout
self, rows: Int, cols: Int):
self.rows = rows
self.cols = cols
# Single contiguous memory allocation
self.data = DTypePointer[DType.float64].alloc(rows * cols)

fn
__del__(owned
self):
self.data.free()


@always_inline


fn
get(self, i: Int, j: Int) -> Float64:
return self.data.load(i * self.cols + j)


@always_inline


fn
set(self, i: Int, j: Int, value: Float64):
self.data.store(i * self.cols + j, value)

fn
matrix_multiply_mojo(borrowed
a: Matrix, borrowed
b: Matrix) raises -> Matrix:
let
rows_a = a.rows
let
cols_b = b.cols

# Single allocation for result
var
result = Matrix(rows_a, cols_b)

for i in range(rows_a):
    for j in range(cols_b):
        var
        sum = 0.0
        # No temporary allocations in this loop
        for k in range(a.cols):
            sum += a.get(i, k) * b.get(k, j)
        result.set(i, j, sum)

return result

# Usage example
fn
main()
raises:
# Create 1000x1000 matrices
var
a = Matrix(1000, 1000)
var
b = Matrix(1000, 1000)

# Initialize matrices (simplified)
for i in range(1000):
    for j in range(1000):
        a.set(i, j, 1.0)
        b.set(i, j, 2.0)

# Perform multiplication with optimized memory usage
let
result = matrix_multiply_mojo(a, b)