highestAngular = 2;
axial[x_,y_,z_, N_] := Flatten/@Table[x^i y^j z^{n-i-j}, {n,0,N}, {i, n, 0, -1}, {j, n-i, 0, -1}];
converter[x, n_] := HoldForm[hx[[n]]];
converter[y, n_] := HoldForm[hy[[n]]];
converter[z, n_] := HoldForm[hz[[n]]];
converter[px, n_] := HoldForm[ax[[n]]];
converter[py, n_] := HoldForm[ay[[n]]];
converter[pz, n_] := HoldForm[az[[n]]];
converter[qx, n_] := HoldForm[bx[[n]]];
converter[qy, n_] := HoldForm[by[[n]]];
converter[qz, n_] := HoldForm[bz[[n]]];

converter[others_, n_] := others^n;
Unprotect[Power];
Format[Power[a_, n_Integer?Positive], CForm] := Distribute[
	ConstantArray[Hold[a], n],
	Hold,List,HoldForm,Times
]
Nfunctions[angular_] := (angular + 1) * (angular + 2) / 2;

(*
axialExpressions = Map[Flatten, Map[ToString[CForm[FullSimplify[#]]]&, 
	Evaluate[
		Expand[Outer[Outer[Times, ##]&, 
			axial[x + px, y + py, z + pz, highestAngular], 
			axial[x + qx, y + qy, z + qz, highestAngular], 1]]/.Power->converter] /. {
				x->HoldForm[hx[[1]]], y->HoldForm[hy[[1]]], z->HoldForm[hz[[1]]],
				px->HoldForm[ax[[1]]], qx->HoldForm[bx[[1]]],
				py->HoldForm[ay[[1]]], qy->HoldForm[by[[1]]],
				pz->HoldForm[az[[1]]], qz->HoldForm[bz[[1]]]
		}
	, {4}], {2}];

densitySnippets = Map[StringRiffle[MapIndexed["result += density["<>ToString[#2[[1]]-1]<>"] * ("<>#1<>")" &, #], ";\n"]<>";"&, axialExpressions, {2}];

densityString = StringRiffle[Flatten@Table[ToString[StringForm["if constexpr (i_angular == `` && j_angular == ``) {", i, j]]<>densitySnippets[[i+1,j+1]] <> "}", {i, 0, highestAngular}, {j, 0, highestAngular}], "\n"];

Print[densityString];
*)

(*
HRRtable = Table[(ToString@StringForm["if constexpr(i_angular==`` && j_angular==``) {``",##])<>"}\n"&@@ {a,b,StringRiffle[#,"\n"]}&@ Flatten[Table[ ToString@StringForm["table[``] = g1 * table[``] + shift * table[``];", ##]&@@(#1 + (a+1) #2&@@@{{i, j}, {i+1, j-1}, {i, j-1}}), {j,1,b},{i,a+b-j,0, -1}]] ,{b,1,highestAngular}, {a,0, highestAngular}];

Print[StringRiffle[Flatten[HRRtable], "\n"]]
*)
RuleX = {
	x1^i_ * x2^j_ :> HoldForm[xij[[aj * j + i]]], 
	x1 x2^j_ :> HoldForm[xij[[(aj+1) * j + 1]]], 
	x1^i_ x2 :> HoldForm[xij[[(aj+1) * 1 + i]]], 
	x1 x2 :> HoldForm[xij[[(aj+1) + 1]]], 
	x1 :> HoldForm[xij[[aj + 1]]], 
	x2 :> HoldForm[xij[[1]]]
}; 

RuleY = {
	y1^i_ * y2^j_ :> HoldForm[yij[[aj * j + i]]], 
	y1 y2^j_ :> HoldForm[yij[[(aj+1) * j + 1]]], 
	y1^i_ y2 :> HoldForm[yij[[(aj+1) * 1 + i]]], 
	y1 y2 :> HoldForm[yij[[(aj+1) + 1]]], 
	y1 :> HoldForm[yij[[aj + 1]]], 
	y2 :> HoldForm[yij[[1]]]
}; 

RuleZ = {
	z1^i_ * z2^j_ :> HoldForm[zij[[aj * j + i]]], 
	z1 z2^j_ :> HoldForm[zij[[(aj+1) * j + 1]]], 
	z1^i_ z2 :> HoldForm[zij[[(aj+1) * 1 + i]]], 
	z1 z2 :> HoldForm[zij[[(aj+1) + 1]]], 
	z1 :> HoldForm[zij[[aj + 1]]], 
	z2 :> HoldForm[zij[[1]]]
}; 

mathForm = Outer[Outer[Times, ##]&, axial[x1,y1,z1, highestAngular],axial[x2,y2,z2, highestAngular], 1];
replacedForm = mathForm/.RuleX /. RuleY /. RuleZ;

Print[StringRiffle[Flatten@Table[ToString@StringForm["if constexpr(ai == `` && aj == ``) ``\n", i-1, j-1, "{" <> StringRiffle[MapIndexed[ToString[StringForm["result += density[``]*``", #2[[1]]-1, #1]]&,ToString/@CForm/@Flatten[replacedForm[[i, j]]]], ";\n"]<>";}\n"], {i, 1, highestAngular + 1},{j,1,highestAngular+1}],"\n"]]

