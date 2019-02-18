## Copyright (C) 2019 Petar Zecevic
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} plot_times (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Petar Zecevic <pzecevic@pzecevic>
## Created: 2019-01-16

function [] = plot_times (n, serial, parallel)
  plot(n, serial, '-r^', n, parallel, '-b^');
  set(gca, "xgrid", "on");
  set(gca, "ygrid", "on");
  #xt = 0:100:int32(n(end))+300;
  #yt = 0:50:int32(serial(end))+300;
  #set(gca, "xtick", xt);
  #set(gca, "ytick", yt);
  ylabel("time[ms]");
  xlabel("neurons per each layer");
  title("Serial and parallel execution time of FFNN4");
endfunction
