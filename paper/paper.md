---
title: 'pypbomb: A Python package with tools for the design of detonation tubes'
tags:
  - detonation
authors:
  - name: Mick Carter
    affiliation: 1
  - name: David Blunck
    affiliation: 1
affiliations:
 - name: School of Mechanical, Industrial, and Manufacturing Engineering, Oregon State University, Corvallis, OR, USA
   index: 1

date: 21 September, 2020
bibliography: paper.bib
---

# Introduction

A detonation is a supersonic combustion reaction in which a reaction front is coupled with a pressure shock front [@Lee2008]. Adiabatic heating from the shocks helps sustain the combustion front, which in turn accelerates the shock front. Because the products of a detonation are at a higher pressure than the reactants, thermodynamic cycles using detonations (such as the Humphrey cycle) have the potential for higher thermodynamic efficiency than deflagration-based cycles (such as the Brayton cycle) [@Coleman2001].

# Statement of need 

In order to study the structure of gaseous detonations, a closed-end detonation tube is required. The design of a detonation tube requires many considerations, including estimation of the required length for deflagration-to-detonation transition (DDT), tube material and size selection (including the effects of transient pressures), fastener failure calculations (including bolt pull-out), flange class selection, viewing window sizing (if optical access is required), and prediction of safe operating conditions (including accounting for detonation reflection). All of this is specific to the mixture being detonated, therefore it is important to be able to quickly re-run the analysis for new mixtures. ``pypbomb`` combines tools for all of these needs in one easy-to-use package.

# Summary

``pypbomb`` contains a series of tools that can be used to quickly design and determine the operational envelope of a closed-end detonation tube. The first iteration of this package was written during the design of just such a tube, which is currently being used to measure cell sizes of gaseous detonations, and will be used in the near future for the study of detonations in two-phase mixtures.

`pypbomb.Tube` allows the user to quickly iterate on the design of the main detonation tube and determine its safe operational limits. Standard pipe size lookups are included, allowing the user to quickly assess different tube geometries. Maximum allowable stress values are available as a function of temperature [@asmeb311], or may be supplied by the user. Using the allowable stress, the maximum pressure of the tube can be calculated [@megyesy]. Shepherd's dynamic load factor is calculated in order to adjust the static pressure limit to account for the tube's response to the transient pressure caused by the detonation wave [@Shepherd2009]. Using the de-rated pressure, the maximum initial reactant pressure can be determined for a given mixture and initial temperature. Functions from the shock and detonation toolbox are used to calculate detonation wave speeds and reflection properties for the desired reactant mixture [@sdt].

Once the operational limits of a tube are determined, flanges can be sized. `pypbomb.Flange` identifies the minimum necessary flange class based on the maximum tube pressure and temperature [@asmeb165].

A successful detonation tube must account for the deflagration-to-detonation transition (DDT). This is usually done using a series of blockages, which cause the combustion wave to undergo local accelerations, thereby aiding in the DDT process [@ciccarelli]. In order to ensure proper detonation is achieved the blockages must be properly sized, and must continue for a minimum (mixture specific) distance. To this end, `pypbomb.DDT` contains tools for Shchelkin spiral blockage ratio and diameter calculations, and allows the user to estimate the necessary DDT run-up length for a desired mixture using Cantera [@ciccarelli; @cantera].

One feature that researchers may desire in a detonation tube is optical access. Historically, the structure of detonations have typically been studied using soot covered foils inserted along the wall or end-cap of detonation tubes [@Lee2008]. More recently, however, researchers have begun using schlieren photography to study detonation waves [@Radulescu2007; @Stevens2015]. In some cases, soot foil and schlieren techniques have been used simultaneously [@Kellenberger2017]. If optical access is desired, window thickness and safety factor calculations are included in `pypbomb.Window` for clamped rectangular windows [@crystran]. Additionally, `pypbomb.Bolt` allows the user to calculate bolt stress areas and safety factors in order to keep the windows intact  and prevent bolts from pulling out of the tube [@machinery]. Standard thread property lookups are included.

# Acknowledgements

This work is supported by the Office of Naval Research, contract N000141612429. The authors would like to thank the Detonation Group at the Air Force Research Laboratory for providing access to a detonation tube, as well as for their invaluable input and technical advice. Additionally, the authors would like to thank Kyle Niemeyer for his instruction and encouragement in the area of open-source software development for engineering research.

# References