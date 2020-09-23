---
title: 'pypbomb: An open-source package with tools for the design of detonation tubes'
tags:
  - detonation
authors:
  - name: Mick Carter
    affiliation: 1
affiliations:
 - name: School of Mechanical, Industrial, and Manufacturing Engineering, Oregon State University, Corvallis, OR, USA
   index: 1

date: 21 September, 2020
bibliography: paper.bib
---

# Introduction

A detonation is a supersonic combustion reaction in which a reaction front is coupled with a pressure shock front. Adiabatic heating from the shocks helps sustain the combustion front, which in turn accelerates the shock front. Because the products of a detonation are at a higher pressure than the reactants, thermodynamic cycles using detonations (such as the Humphrey cycle) have the potential for higher thermodynamic efficiency than deflagration-based cycles (such as the Brayton cycle) [@Coleman2001].

# Statement of need 

In order to study the structure of gaseous detonations, a closed-end detonation tube is required. The design of a detonation tube requires many considerations, including estimation of the required length for deflagration-to-detonation transition (DDT), tube material and size selection (including the effects of transient pressures), fastener failure calculations (including bolt pull-out), flange class selection, viewing window sizing (if optical access is required), and prediction of safe operating conditions (including accounting for detonation reflection). All of this is specific to the mixture being detonated, therefore it is important to be able to quickly re-run the analysis for new mixtures. ``pypbomb`` combines tools for all of these needs in one easy-to-use package.

# Summary

``pypbomb`` contains a series of tools that can be used to quickly design and determine the operational envelope of a closed-end detonation tube. The first iteration of this package was written during the design of just such a tube, which is currently being used to measure cell sizes of gaseous detonations, and will be used in the near future for the study of detonations in two-phase mixtures.

overview

## Tube

* max allowable stress lookup as f(T) [@asmeb311]
* max allowable pressure calculation [@megyesy]
* dynamic load factor calculation [@Shepherd2009]
* max safe initial pressure calculation [@sdt]
* standard pipe size lookup and relevant material property lookup for stainless steels
* pipe dimensions and stress limits (welded and seamless)

## Flange

* Properly size flanges using max tube pressure at a given temperature [@asmeb165]

## DDT

* Shchelkin spiral blockage ratio and diameter calculations
* DDT run-up length estimation [@ciccarelli]
  * BR <= 0.1 may require phase specification for cantera [@cantera]

## Window

Determine required window thickness and safety factor for clamped rectangular windows [@crystran].

## Bolt

* calculate bolt stress areas [@machinery]
* calculate bolt safety factor
* thread property lookup

## Acknowledgements

This work is supported by the Office of Naval Research, contract N000141612429. The author would like to thank the Detonation Group at the Air Force Research Laboratory for providing access to a detonation tube, as well as for their invaluable input and technical advice. Additionally, the author would like to thank Kyle Niemeyer for his instruction and encouragement in the area of open-source software development for engineering research.

# References