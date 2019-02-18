## Copyright (C) 2018 Petar Zecevic
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
## @deftypefn {} {@var{retval} =} neuralNetworkOutput (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Petar Zecevic <pzecevic@pzecevic>
## Created: 2018-12-28

function [output] = neuralNetworkOutput (input, wMatrices, actfunction)
# input is vector column.
# wMatrices is a cell.
# output is vector row.

[l_num,~] = size(wMatrices);
out_size = length(input);
output = zeros(out_size,1);
output = actfunction(input);

for i = [1:l_num]
  output = actfunction(wMatrices{i}*output);
endfor

endfunction
