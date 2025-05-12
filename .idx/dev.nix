# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "unstable"; # Using unstable for better package availability

  # Use https://search.nixos.org/packages to find packages
  packages = [
    # Existing packages
    pkgs.nodejs_20
    pkgs.util-linux
    pkgs.docker
    pkgs.docker-compose
  ];

  # Sets environment variables in the workspace
  env = {
    # Added environment variables for future use
    AWS_ACCESS_KEY_ID = "";
    AWS_SECRET_ACCESS_KEY = "";
    SOLANA_KEYPAIR_PATH = "/path/to/solana-keypair.json";
    ETHEREUM_PRIVATE_KEY = "";
    DATABASE_URL = "postgresql://user:password@localhost:5432/akshuai";
    REDIS_URL = "redis://localhost:6379";
  };

  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "vscodevim.vim"
      "golang.go"
      "dbaeumer.vscode-eslint"
      "esbenp.prettier-vscode"
      "ms-vscode.vscode-typescript-next"
      "rust-lang.rust-analyzer"
      "ms-python.python"
      "ms-toolsai.jupyter"
      "nomicfoundation.hardhat-solidity"
      "redhat.vscode-yaml"
      "ms-azuretools.vscode-docker"
      "hashicorp.terraform"
      "amazonwebservices.aws-toolkit-vscode"
    ];

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        npm-install = "yarn install --frozen-lockfile --prefer-offline --no-progress";
        default.openFiles = [ "README.md" "index.ts" "pages/index.js" ];
        # Initialize PostgreSQL database
        init-postgres = ''
          if ! pg_isready -q; then
            echo "Initializing PostgreSQL..."
            initdb -D /var/lib/postgresql/data
            pg_ctl -D /var/lib/postgresql/data -l logfile start
            createdb akshuai
          fi
        '';
        # Initialize Redis
        init-redis = ''
          if ! redis-cli ping; then
            echo "Starting Redis..."
            redis-server --daemonize yes
          fi
        '';
        # Install Python dependencies for AI/ML
        install-python-deps = ''
          pipenv install tensorflow torch jupyter --skip-lock
          pipenv run jupyter notebook --generate-config
        '';
      };

      # Runs when the workspace is (re)started
      onStart = {
        run-server = ''
          if [ -z "$GOOGLE_GENAI_API_KEY" ]; then
            echo 'No Gemini API key detected. Enter one from https://aistudio.google.com/app/apikey:'
            read -s GOOGLE_GENAI_API_KEY
            echo 'You can also add it to .idx/dev.nix to auto-load it next time.'
            export GOOGLE_GENAI_API_KEY
          fi

          # Start PostgreSQL and Redis if not running
          pg_ctl -D /var/lib/postgresql/data -l logfile start || echo "PostgreSQL already running"
          redis-server --daemonize yes || echo "Redis already running"

          # Ensure Solana CLI is in PATH
          export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"

          # Start the Next.js development server
          yarn dev
        '';
      };
    };
  };
}