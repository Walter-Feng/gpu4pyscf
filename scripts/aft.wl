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
