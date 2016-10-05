function make(target,flags)
% MAKE Simulates GNU-make functionality and facilitates mex-file compilation.
%   Edit make.m to set the source and build directories, include and link
%   libraries and define the source files to be compiled into mex files.
%
%   MAKE all or MAKE('all') compiles all mex files.
% 
%   MAKE all -flags or MAKE('all',['-' flags]) allows user input to control
%   debug, warnings, and verbose flags. flags can either be a single word
%   or a sequence of flags.
% 
%   Examples:
%   make all -debug or make('all','-debug'): compiles using debug symbols
%   make all -warnings or make('all','-warnings'): prints all warnings
%   make all -verbose or make('all','-verbose'): prints verbose information
% 
%   You can use a combination of flags in the following way:
%   make all -dw:   compiles using debug symbols and prints all warnings
%   make all -v:    be verbose
%   make all -wvd:  set debug, warnings and verbose flags to true
% 
% NOTE: make.m assumes that you are using the gcc compiler. If you plan to
% use a different compiler, you might have to change some of the flags.
% 
% Stavros Tsogkas, <stavros.tsogkas@centralesupelec.fr>
% Last update: March 2015 


narginchk(0,2);
if nargin < 1, target = 'all'; end
if nargin < 2
    debug = false; verbose = false; warnings = false;
else
    [debug,verbose,warnings] = parseFlags(flags);
end

% Source files to be compiled and build directory
sourceDir   = 'src';      % your source files are here
buildDir    = 'build';    % the mex file will be placed here
sourceFiles = readSourceFiles(sourceDir);

% Source files to be compiled into mex files (add more at will)
mexFiles    = {'examples/denseInferenceMex.cpp','examples/denseInferenceMexEdge.cpp'}; 

% Include directories and Link libraries
INCLUDEDIRS = {'include','external','examples','include/Eigen3.2.4/'}; %,'/usr/local/include/'} ;
%,...
%    '/Users/Shared/Eigen3.2.4',...
%    '/Users/Shared/LBFGS/include'};
%LINKLIBS    = {'/Users/Shared/LBFGS/lib/ -llbfgs'};
%LINKLIBS    = {'/usr/local/lib/ -llbfgs'};
LINKLIBS = {''};
% Optimization flags (remove or change at will)
if 0
CXXOPTIMFLAGS = ' CXXOPTIMFLAGS="-O3 -DNDEBUG -fopenmp -march=native -ffast-math -pthread -pipe -msse2"';
LDOPTIMFLAGS  = ' LDOPTIMFLAGS="-O3 -DNDEBUG -fopenmp -march=native -ffast-math -pthread -pipe -msse2"';
else
CXXOPTIMFLAGS = ' CXXOPTIMFLAGS="-O3 -DNDEBUG  -march=native -ffast-math -pthread -pipe -msse2"';
LDOPTIMFLAGS  = ' LDOPTIMFLAGS="-O3 -DNDEBUG  -march=native -ffast-math -pthread -pipe -msse2"';
end
CXXFLAGS      = ' CXXFLAGS="\$CXXFLAGS -Wall"';
LDFLAGS       = ' LDFLAGS="\$LDFLAGS -Wall"';

switch target
    case 'all'
        % Build mex command
        mexcmd = 'mex';
        if debug
            mexcmd = [mexcmd ' -g'];
        else
            mexcmd = [mexcmd ' -O' CXXOPTIMFLAGS LDOPTIMFLAGS];
        end
        if verbose,  mexcmd = [mexcmd ' -v']; end
        if warnings, mexcmd = [mexcmd CXXFLAGS LDFLAGS]; end
        if isdir(buildDir), mexcmd = [mexcmd ' -outdir ' buildDir]; end
        mexcmd = [mexcmd ' -largeArrayDims'];
        mexcmd = addIncludeDirs(mexcmd, INCLUDEDIRS);
        %mexcmd = addLinkLibs(mexcmd, LINKLIBS);
        mexcmd = addSourceFiles(mexcmd, sourceFiles);
        disp(['Creating ' buildDir 'directory']); mkdir(buildDir);
        mexcmd
        buildMexFiles(mexFiles, mexcmd);
        %movefile('build/denseInferenceMex.mexa64','.')
        %!rmdir build
        
    case 'clean'
        deleteMexFiles(buildDir, mexFiles)
    otherwise
        error('Invalid make target')
end

function buildMexFiles(mexFiles, mexcmd)
for i=1:numel(mexFiles)
    [~,mexName] = fileparts(mexFiles{i}); 
    mexcmdFull  = [mexcmd ' ' mexFiles{i} ' -output ' mexName];
    disp(['Executing: ' mexcmdFull]); eval(mexcmdFull);
end

function mexcmd = addIncludeDirs(mexcmd, INCLUDEDIRS)
for i=1:numel(INCLUDEDIRS)
    mexcmd = [mexcmd ' -I' INCLUDEDIRS{i}];
end

function mexcmd = addLinkLibs(mexcmd, LINKLIBS)
for i=1:numel(LINKLIBS)
    mexcmd = [mexcmd ' -L' LINKLIBS{i}];
end

function mexcmd = addSourceFiles(mexcmd, sourceFiles)
%TODO: check if file exists already and if it has been changed 
for i=1:numel(sourceFiles)
    mexcmd = [mexcmd ' ' sourceFiles{i}];
end

function deleteMexFiles(buildDir, sourceFiles)
for i=1:numel(sourceFiles)
    [~,name] = fileparts(sourceFiles{i});
    mexFile  = [name '.' mexext];
    delete(fullfile(buildDir, mexFile));
end

function sourceFiles = readSourceFiles(sourceDir)
sourceFiles = [dir([sourceDir '/*.cpp']); 
               dir([sourceDir '/*.cc']);
               dir([sourceDir '/*.c'])];
sourceFiles = {sourceFiles(:).name}; 
for i=1:numel(sourceFiles)
    sourceFiles{i} = fullfile(sourceDir, sourceFiles{i});
end

function [debug,verbose,warnings] = parseFlags(flags)
%TODO: better flag checking (assert that input flags are valid etc)
assert(flags(1)=='-', 'Invalid syntax. Flags should be preceded by ''-''');
flags    = flags(2:end);
debug    = ismember(flags, {'debug','Debug','DEBUG'}) || ...
    ~isempty(strfind(flags,'d')) || ~isempty(strfind(flags,'D'));
verbose  = ismember(flags, {'verbose','Verbose','VERBOSE'}) || ...
    ~isempty(strfind(flags,'v')) || ~isempty(strfind(flags,'V'));
warnings = ismember(flags, {'warnings','Warnings','WARNINGS'}) || ...
    ~isempty(strfind(flags,'w')) || ~isempty(strfind(flags,'W'));
