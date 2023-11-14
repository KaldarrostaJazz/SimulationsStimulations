#include <Python.h>
#include <numpy/arrayobject.h>

extern "C" {
    // Define your C++ function that takes NumPy arrays as arguments
    void your_function(PyObject* input_array) {
        // Convert the input PyObject to a NumPy array
        PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(input_array);

        // Access and manipulate the NumPy array data
        // Example: Accessing the dimensions and data
        int ndim = PyArray_NDIM(np_array);
        npy_intp* dims = PyArray_DIMS(np_array);
        // ... Further processing of the NumPy array

        // Don't forget to release the NumPy array
        Py_DECREF(np_array);
    }
}
