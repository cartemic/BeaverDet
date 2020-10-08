Changelog
---
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2018-06-12 - Pre-release
* Initial package developed as ``BeaverDet``  in ME-599
* Required ``SD2``, since at the time ``sdtoolbox`` was unavailable for Python 3

## [1.0.0] - 2020-09-21 - Initial Release
* Lots of streamlining
* There was an attempt to make things too automatic, which was rolled back
* ``SD2`` mostly removed, but some functions have been incorporated into
  ``pypbomb`` because:
  * My adaptation of the CJ speed calculation allows for parallelization during
    the curve fit, which can speed things up considerably
  * ``sdtoolbox`` still needs to be manually installed and is not on github 
    (although I think I've mostly figured out how to hack it to self-install)
* Initial documentation has been added

## [1.1.0] - 2020-10-?? (not released yet)
* Changelog is now actually being updated
* Main classes (``Tube``, ``DDT``, etc.) have been imported at the top level of the package for ease of use
* Documentation has been improved
* Package can now be installed via ``setup.py``
