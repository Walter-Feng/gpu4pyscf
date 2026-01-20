axial[x_,y_,z_, N_] := Flatten/@Table[x^i y^j z^{n-i-j}, {n,0,N}, {i, n, 0, -1}, {j, n-i, 0, -1}];
