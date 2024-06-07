#!/bin/bash
set -e -x

for PYBIN in /opt/python/cp3[89]*/bin /opt/python/cp310*/bin; do
    "${PYBIN}/pip" install maturin
    "${PYBIN}/maturin" build -i "${PYBIN}/python" --release
done

mkdir -p wheelhouse
for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}" -w wheelhouse/
done

# Move repaired wheels back to target/wheels
mv wheelhouse/*.whl target/wheels/

/opt/python/cp310-cp310/bin/maturin publish --username __token__ --password ${PYPI_API_TOKEN}