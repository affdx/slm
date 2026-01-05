interface LogoProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

const sizes = {
  sm: { width: 100, height: 27 },
  md: { width: 150, height: 40 },
  lg: { width: 300, height: 80 },
};

export function Logo({ className, size = "md" }: LogoProps) {
  const { width, height } = sizes[size];

  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 300 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Isyarat"
      role="img"
    >
      <g fill="currentColor">
        <path d="M28,60 L12,60 L12,30 C12,30 10,20 20,10 C30,20 28,30 28,30 L28,60 Z M12,62 L28,62 L26,75 L14,75 L12,62 Z" />
        <text
          x="35"
          y="75"
          fontFamily="'Arial Black', 'Helvetica Neue Bold', sans-serif"
          fontWeight="900"
          fontSize="64"
          letterSpacing="-1"
        >
          SYARAT
        </text>
      </g>
    </svg>
  );
}

export function LogoIcon({ className }: { className?: string }) {
  return (
    <svg
      width="32"
      height="32"
      viewBox="0 0 40 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <g fill="currentColor">
        <path d="M28,60 L12,60 L12,30 C12,30 10,20 20,10 C30,20 28,30 28,30 L28,60 Z M12,62 L28,62 L26,75 L14,75 L12,62 Z" />
      </g>
    </svg>
  );
}
