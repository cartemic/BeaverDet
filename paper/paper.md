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

## Tube

`pypbomb.tube.Tube` allows the user to quickly iterate on the design of the main detonation tube and determine its safe operational limits. Standard pipe size lookups are included, allowing the user to quickly assess different tube geometries. Maximum allowable stress values are available as a function of temperature [@asmeb311], or may be supplied by the user. Using the allowable stress, the maximum pressure of the tube can be calculated [@megyesy]. Shepherd's dynamic load factor is calculated in order to adjust the static pressure limit to account for the tube's response to the transient pressure caused by the detonation wave [@Shepherd2009]. Using the de-rated pressure, the maximum initial reactant pressure can be determined for a given mixture and initial temperature. The shock and detonation toolbox is used to calculate detonation wave speeds and reflection properties [@sdt, @cantera].

## Flange

`pypbomb.tube.Flange` identifies the necessary flange class based on the maximum tube pressure and temperature [@asmeb165].

## DDT

A successful detonation tube will allow a successful deflagration-to-detonation transition (DDT). `pypbomb.tube.DDT` contains tools for Shchelkin spiral blockage ratio and diameter calculations. It also allows the user to estimate the DDT run-up length for a desired mixture using cantera [@ciccarelli, @cantera].

## Window

For optical access, window thickness and safety factor calculations are included in `pypbomb.tube.Window` for clamped rectangular windows [@crystran].

## Bolt

`pypbomb.tube.Bolt` allows the user to calculate bolt stress areas and safety factors in order to keep the windows intact  and prevent bolts from pulling out of the tube [@machinery]. Standard thread property lookups are included.

## Acknowledgements

This work is supported by the Office of Naval Research, contract N000141612429. The author would like to thank the Detonation Group at the Air Force Research Laboratory for providing access to a detonation tube, as well as for their invaluable input and technical advice. Additionally, the author would like to thank Kyle Niemeyer for his instruction and encouragement in the area of open-source software development for engineering research.

# References