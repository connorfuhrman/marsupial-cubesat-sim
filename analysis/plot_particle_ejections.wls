#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*TrimDataPoint[d_]:= d[[2;;All]];*)
(*ConditionDataPoint[d_]:= Divide[#, 1*^3]& /@ TrimData[d];*)
(*ConditionEjectionEvent[e_]:= ConditionDataPoint /@ e;*)
(*ConditionDataset[s_]:=ConditionEjectionEvent /@ s;*)
(*datadir = "/Users/connorfuhrman/Library/Mobile Documents/com~apple~CloudDocs/UArizona/SpaceTREx/projects/decentralized_vlc_sample_return/repo/bennu_particles/generated/";*)
(*alldata = Import[#]& /@ FileNames["*.csv", datadir];*)
(**)


(* ::Input:: *)
(*ListPointPlot3D[ConditionDataset@alldata,*)
(*			PlotTheme->"Scientific",*)
(*			PlotStyle->Blue,*)
(*			PlotRange->Table[{-1,1},3],*)
(*			FrameLabel]*)
