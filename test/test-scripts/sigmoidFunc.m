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

## -*- Calculate sigmoid activation function -*- 
## @deftypefn {} {@var{y} =} sigmoidFunc (@var{x})
##
## @seealso{}
## @end deftypefn

## Author: Petar Zecevic <pzecevic@pzecevic>
## Created: 2018-12-25

function [y] = sigmoidFunc (x)
  y = 1 ./ (1 + exp(-x));
endfunction
