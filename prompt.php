<!DOCTYPE html>
<html>

<head>
    <title>Recherche</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
        }

        form {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
            margin: 50px auto;
            padding: 20px;
            max-width: 600px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            border-radius: 5px;
        }

        input[type=text] {
            width: 100%;
            padding: 12px 20px;
            margin: 8px 0;
            box-sizing: border-box;
            border: none;
            border-bottom: 2px solid #ddd;
            font-size: 16px;
            outline: none;
            background-color: #fff;
        }

        button[type=submit] {
            background-color: #3E83C4;
            color: #fff;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }

        button[type=submit]:hover {
            background-color: #3e6fc4;
        }

        .title-container {
            text-align: center;
            margin-top: 50px;
        }

        h1 {
            font-size: 36px;

            font-weight: bold;
            margin: 0;
            color: #3E83C4;
        }

        .result {
            margin: 20px 0;
            padding: 10px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
            border-radius: 5px;
        }

        h2 {
            font-size: 24px;
            font-weight: bold;
            margin: 0 0 10px;
        }

        p {
            font-size: 16px;
            margin: 0;
        }
    </style>
</head>

<body>
    <div class="title-container">
        <h1 style="color: #3E83C4;">Recherche</h1>
    </div>
    <form action="" method="post">
        
        <label for="prompt">Prompt :</label>
        <input type="text" name="prompt" placeholder="prompt"><br>
        <button type="submit" name="submit" value="submit">Search</button>
    </form>





    

<?php


$inputSet = array();

$ok = false;

if(isset($_POST['submit'])){

    if(isset($_POST['prompt']) && $_POST['prompt'] != ''){
        $inputSet['Prompt'] = $_POST['prompt'];
    }

    $ok = !empty($inputSet);


}

if(!$ok){

    echo '<br>En attente de prompt <br>';

}else{

    // Echapper la phrase pour des raisons de securite
    $phrase_escaped = escapeshellarg($inputSet['Prompt']);

    // Appeler le script Python avec la phrase en tant qu'argument
    $output = shell_exec("python3 projet_f.py $phrase_escaped");

    $outputArray = explode(",", $output);

    // Afficher la sortie du script Python
    foreach($outputArray as $code) {
        $href = "https://plateforme.kerdos-energy.com/index.php#/search/" . $code;
        echo "Le code est : K" . $code . ", Lien :" . '<a href=' . $href  . '>Cliquez ici pour la solution</a>' . "<br>";
    }

}

?>




</body>

</html>