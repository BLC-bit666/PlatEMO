classdef QRImage_BC < ReproducedApplication
% <2026> <multi> <real> <constrained> <expensive>
% Fixed sunflower QR design: actual JPEG bytes and image MSE.
% D=17: sixteen mixing coefficients and one JPEG-quality control, all [0,1].
% C=0 iff all three unchanged ZXing-C++ checks decode the required payload.
    methods
        function Setting(obj)
            obj.configure('qr',17,1);
        end
    end
end
