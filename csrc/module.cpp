#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_C",
    NULL,
    -1,
    NULL, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__C(void)
{
    return PyModule_Create(&moduledef);
}