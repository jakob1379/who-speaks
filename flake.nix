{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      runtimeLibs = with pkgs; [
        glib
        libGL
        libGLU
        libice
        libxcb
        libsm
        libxkbcommon
        libx11
        libxau
        libxdmcp
        libxext
        libxfixes
        libxi
        libxrandr
        libxrender
        libxtst
        libxcb-util
        libxcb-image
        libxcb-keysyms
        libxcb-render-util
        libxcb-wm
        stdenv.cc.cc.lib
        zlib
      ];
    in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            ffmpeg
            python312
            uv
          ];

          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH";
          QT_QPA_PLATFORM = "offscreen";
        };
      }
  );

}
