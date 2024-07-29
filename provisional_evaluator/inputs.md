# TDOAH and Foy Solvers
Requirements.txt might not working properly, so here are the packages you need to install:
```bash
json
numpy
math
```

## Contents of `evaluation_functions.py`
```python
x, y, z = w2k(lat, long, h)
```
Converts WGS84 coordinates to Cartesian coordinates.

```python
lat, long, alt = k2w(x, y, z)
```
Converts Cartesian coordinates to WGS84 coordinates.

```python
delta_h = pressure2height(pressureRef, pressure)
```
Converts pressure difference to height difference.






### Tdoah Class:
```python
solver = Tdoah(bs, ms, ms_h, tdoa01, tdoa02)
```

`bs`: [[float, float, float], [float, float, float], [float, float, float], _[float, float, float]_] # Last coordinate is ms

`ms`: [float, float, float]

```python
target = solver.solve(ms_h, tdoa01, tdoa02)
```

`ms_h`: **float** in meters

`tdoa01`: **float** 

`tdoa02`: **float**

### Foy Class:
```python
solver = Foy(bs, ms, tdoa01, tdoa02, tdoa03)
```

`bs`: [[float, float, float], [float, float, float], [float, float, float], _[float, float, float]_] # Last coordinate is ms

`ms`: [float, float, float]

```python
solver.solve(tdoa01, tdoa02, tdoa03)
```

`tdoa01`: **float** 

`tdoa02`: **float**

`tdoa03`: **float**

```python
target = solver.guesses[0][-1], solver.guesses[1][-1], solver.guesses[2][-1]
```

TODOS:
- [x] Make solver take tdoas as input
- [ ] Altitude or pressure as input
- [ ] Make solver take baro as input
- [x] Fix Foy solver
- [x] Cleanup solver call (inputs)
- [x] Implement function to calculate altitude from pressure