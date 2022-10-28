#!/usr/bin/env wolframscript


argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;

ShowHelp[] := (
	Print["ERROR - Not Enough Command Line Arguments: ",
	      "Call as './plot_simulation.wsl <path to experiment results> [<path> ...]'"];
	Exit[1]; );

If[argc < 1, ShowHelp[]];

(* Get associations from our metadata *)
GetData[fname_] := Module[{f = fname, json, mdata},
			  json = Import[fname, "RawJSON"];
			  mdata = json[["metadata"]];
			  <|"num_crafts" -> mdata[["max_num_crafts"]], 
			    "num_samples" -> mdata[["max_num_samples"]], 
			    "sim_time" -> mdata[["sim_time"]], 
			    "elapsed_time" -> mdata[["time_end"]] - mdata[["time_start"]]|>
		   ];


(* Get all files matching *.json in all subdirectories of `toplevel` *)
GetJSONFiles[toplevel_] := FileNames["*.json", toplevel, Infinity];


jsonfiles = Flatten[GetJSONFiles /@ argv];
Print["Gathering ", Length[jsonfiles], " simulation results"];
simdata = GetData /@ jsonfiles;

imgsize = {800, 600}

Export["num_crafts-v-sim_time.png", 
       ListLinePlot[
	       Values[simdata[[All, {"num_crafts", "sim_time"}]]],
	       Joined->False,
	       PlotLabel->"Number of Crafts vs Sim Time",
	       FrameLabel->{"Number of crafts", "Simulation time [sec]"},
	       PlotTheme->"Scientific",
	       ImageSize->imgsize],
       ImageResolution->imgsize
]

Export["sim_time-v-elapsed_time.png", 
       ListLinePlot[
	       Values[simdata[[All, {"sim_time", "elapsed_time"}]]],
	       Joined->False,
	       PlotLabel->"Simulation Time vs Elapsed Time",
	       FrameLabel->{"Simulation time [sec]", "Elapsed time [sec]"},
	       PlotTheme->"Scientific",
	       ImageSize->imgsize],
       ImageResolution->imgsize
]

Export["num_crafts-v-elapsed_time.png",
       ListLinePlot[
	       Values[simdata[[All, {"num_crafts", "elapsed_time"}]]],
	       Joined->False,
	       PlotLabel->"Number of crafts vs Elapsed Time",
	       FrameLabel->{"Number of crafts", "Elapsed time [sec]"},
	       PlotTheme->"Scientific",
	       ImageSize->imgsize],
       ImageResolution->imgsize
]

Print["Plots are in your cwd"]
