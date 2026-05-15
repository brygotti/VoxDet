## Prepare environment

```
pip install mayavi==4.8.1
pip uninstall -y vtk
pip install --extra-index-url https://gitlab.kitware.com/api/v4/projects/13/packages/pypi/simple vtk-osmesa==9.3.1
pip install PyQt5
pip install configobj
```

### Visualization

```shell
python tools/visualize.py
```

