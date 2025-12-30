highestAngular = 2;
axial[x_,y_,z_, N_] := Flatten/@Table[x^i y^j z^{n-i-j}, {n,0,N}, {i, n, 0, -1}, {j, n-i, 0, -1}];
converter[x, n_] := Hx[n];
converter[y, n_] := Hy[n];
converter[z, n_] := Hz[n];
converter[others_, n_] := others^n;
Unprotect[Power];
Format[Power[a_, n_Integer?Positive], CForm] := Distribute[
	ConstantArray[Hold[a], n],
	Hold,List,HoldForm,Times
]
arrayFormatter[string_] := StringReplace[string, {")"->"]", "("->"["}]

(* Note that gx[0] should refer to gx, rather than 1 in actual Hermite polynomial definition *)
Print[Map[arrayFormatter[ToString[CForm[#]]]&, 
	Evaluate[
		Expand[Outer[Outer[Times, ##]&, 
			axial[x + x1[0], y + x1[1], z+x1[2], highestAngular], 
			axial[x + x2[0], y + x2[1], z+x2[2], highestAngular], 1][[1, 3]]]/.Power->converter] /. {x->gx,y->gy,z->gz}
	, {2}]]


