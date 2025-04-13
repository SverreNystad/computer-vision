let
    pkgs = import <nixpkgs> {
       config = {
           allowUnfree = true;  # cuDNN is an unfree package, so this is needed
           cudaSupport = true;  # Enable CUDA support
           cudnnSupport = true; # Enable cuDNN support
       };
    };
    lib = pkgs.lib;
    ld_packages = [
        pkgs.libxkbcommon
        pkgs.libGL
        pkgs.wayland
    ];
in pkgs.mkShell {
        packages = [
            (pkgs.python3.withPackages (python-pkgs: [
                python-pkgs.numpy
                python-pkgs.matplotlib
                python-pkgs.torch-bin
                python-pkgs.torchvision-bin
            ])
            )
            pkgs.cudaPackages.cudatoolkit
            pkgs.libtorch-bin
        ];

        LD_LIBRARY_PATH = lib.makeLibraryPath ld_packages;


        shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LIBTORCH_LIB=${pkgs.libtorch-bin}/lib
    export LIBTORCH=${pkgs.libtorch-bin}/lib
    export LIBTORCH_INCLUDE=${pkgs.libtorch-bin.dev}
    echo "Use nixGL python ... for python to recognize the GPU/CUDA and so on"
    '';
    }
