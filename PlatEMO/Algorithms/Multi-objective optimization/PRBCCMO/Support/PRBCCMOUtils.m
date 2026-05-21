classdef PRBCCMOUtils
% Utility helpers shared by PRBCCMO tracing, summary, and analysis scripts.

    methods (Static)
        function requireColumns(T,Names,filePath)
            if nargin < 3
                filePath = "";
            end
            Missing = setdiff(string(Names),string(T.Properties.VariableNames));
            if strlength(string(filePath)) > 0
                assert(isempty(Missing), ...
                    'PRBCCMOUtils:MissingColumns', ...
                    'Missing columns in %s: %s', char(string(filePath)), strjoin(cellstr(Missing), ', '));
            else
                assert(isempty(Missing), ...
                    'PRBCCMOUtils:MissingColumns', ...
                    'Missing columns: %s', strjoin(cellstr(Missing), ', '));
            end
        end

        function Flag = hasColumns(T,Names)
            Flag = all(ismember(cellstr(string(Names)),T.Properties.VariableNames));
        end

        function T = numericizeTable(T)
            Names = T.Properties.VariableNames;
            for i = 1 : numel(Names)
                Value = T.(Names{i});
                if iscell(Value)
                    Value = string(Value);
                end
                if isstring(Value)
                    Num = str2double(Value);
                    if any(isfinite(Num))
                        T.(Names{i}) = Num;
                    else
                        T.(Names{i}) = Value;
                    end
                elseif islogical(Value)
                    T.(Names{i}) = double(Value);
                end
            end
        end

        function value = scalarValue(T,Name)
            value = double(T.(Name)(1));
        end

        function value = valueAt(T,Name,idx,Default)
            if nargin < 4
                Default = NaN;
            end
            if isnan(idx) || ~PRBCCMOUtils.hasColumns(T,{Name})
                value = Default;
            else
                value = double(T.(Name)(idx));
            end
        end

        function value = valueAtString(T,Name,Default)
            if PRBCCMOUtils.hasColumns(T,{Name})
                value = string(T.(Name)(1));
            else
                value = string(Default);
            end
        end

        function value = metaString(Meta,Name,Default)
            if PRBCCMOUtils.hasColumns(Meta,{Name}) && height(Meta) > 0
                value = string(Meta.(Name)(1));
            else
                value = string(Default);
            end
        end

        function value = metaDouble(Meta,Name,Default)
            value = Default;
            if PRBCCMOUtils.hasColumns(Meta,{Name}) && height(Meta) > 0
                Raw = Meta.(Name);
                Parsed = str2double(string(Raw(1)));
                if isfinite(Parsed)
                    value = Parsed;
                elseif isnumeric(Raw) || islogical(Raw)
                    value = double(Raw(1));
                end
            end
        end

        function value = meanFinite(values)
            values = PRBCCMOUtils.finiteValues(values);
            if isempty(values)
                value = NaN;
            else
                value = mean(values);
            end
        end

        function value = stdFinite(values)
            values = PRBCCMOUtils.finiteValues(values);
            if numel(values) < 2
                value = NaN;
            else
                value = std(values,0);
            end
        end

        function value = medianFinite(values)
            values = PRBCCMOUtils.finiteValues(values);
            if isempty(values)
                value = NaN;
            else
                value = median(values);
            end
        end

        function value = sumFinite(values)
            values = PRBCCMOUtils.finiteValues(values);
            if isempty(values)
                value = 0;
            else
                value = sum(values);
            end
        end

        function value = percentileFinite(values,p)
            values = sort(PRBCCMOUtils.finiteValues(values));
            if isempty(values)
                value = NaN;
            else
                idx = max(1,min(numel(values),ceil(p/100*numel(values))));
                value = values(idx);
            end
        end

        function rate = positiveRate(values)
            values = PRBCCMOUtils.finiteValues(values);
            if isempty(values)
                rate = NaN;
            else
                rate = mean(values > 0);
            end
        end
    end

    methods (Static, Access = private)
        function values = finiteValues(values)
            values = double(values);
            values = values(isfinite(values));
        end
    end
end
