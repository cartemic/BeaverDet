# OVERALL:
    - add unit conversion classes
    - change all classes to functions of a new det tube object
    - improve commenting
    - add unit testing to each function and class
    - debug and ship modes etc.
    - remove blank except blocks
    - incorporate known gas mixtures
    - be slightly less mean to the user
    - PIE IN THE SKY: GUI

## pipe():
    - incorporate unit conversions
    - change from class to function
    - externalize allowable stress/temp lookup
    - change from class to function
### getPipe(): 
    - externalize schedule/OD/thickness lookup
    - remove blank except
### getDLF():
    - remove blank except
### getPMax():
    - remove blank except
### info():
    - lump into main class info function
    - remove blank except

## window():
    - incorporate unit conversions
    - change from class to function
    - external material lookup by name instead of rupture modulus input
    - fix sympy problems on recalculate between circular/rectangular geometry
    - lump info into main class info function and remove blank except

## spiral():
    - remove struts
    - change from class to function
### get_spiral_diameter():
    - add documentation
    - remove blank except
### get_blockage_ratio():
    - remove all strutness
### info():
    - lump into main class info function
    - remove blank except

## flange():
    - incorporate unit conversions
    - externalize temperature/pressure/class lookup
    - improve commenting
    - change from class to function
    - remove blank except

## reflection():
    - add documentation
    - incorporate unit conversions
    - look into using the reflection function in SDToolbox
    - lump info into main class info function

## bolt_pattern():
    - incorporate unit conversions
    - be more explicit about the fact that this is for the window bolts
    - externalize thermal derating lookup

