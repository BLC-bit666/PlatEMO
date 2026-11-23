classdef BlackForestRoute_BC < ReproducedApplication
% <2026> <multi> <real> <constrained> <expensive>
% Fixed Black Forest route preferences: total km and cycling ascent metres.
% D=5, each [0,100]; fixed endpoints and two pinned real-map tiles.
% C=0 iff every generated route edge has mode=bike; mixed routes retain objectives.
    methods
        function Setting(obj)
            obj.configure('route',5,100);
        end
    end
end
